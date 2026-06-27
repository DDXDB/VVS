import os
import sys
import time

# Relax allocation limits for some runtimes
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"

import gradio as gr
import librosa
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
from tqdm import tqdm

# ============================================================================
# 设备与ATTN管理模块
# ============================================================================

class AttentionManager:
    """智能Attention选择器，支持自动优先级和回退机制"""

    # 为每个设备定义Attention优先级（从高到低）
    ATTN_PRIORITIES = {
        "cuda": ["flash_attention_2", "flash_attn2", "flash_attn", "sdpa", "eager"],
        "xpu": ["flash_attention_2", "flash_attn2", "flash_attn", "sdpa", "eager"],
        "mps": ["sdpa", "eager"],
        "cpu": ["eager"],
    }

    # 包与实现的映射
    PACKAGE_TO_IMPL = {
        "flash_attention": "flash_attention_2",
        "flash_attn2": "flash_attn2",
        "flash_attn": "flash_attn",
    }

    @staticmethod
    def check_package(pkg_name: str) -> bool:
        """检查包是否可用"""
        try:
            __import__(pkg_name)
            return True
        except ImportError:
            return False

    @staticmethod
    def get_available_impls(device: str) -> dict:
        """获取指定设备可用的Attention实现"""
        available = {}
        for impl in AttentionManager.ATTN_PRIORITIES.get(device, ["eager"]):
            for pkg, pkg_impl in AttentionManager.PACKAGE_TO_IMPL.items():
                if pkg_impl == impl and AttentionManager.check_package(pkg):
                    available[impl] = f"({pkg})"
                    break
            else:
                if impl in ("eager", "sdpa"):
                    available[impl] = "(built-in)"
        return available

    @staticmethod
    def select_attn(device: str) -> str:
        """选择最优的Attention实现"""
        env_attn = os.environ.get("ATTN_IMPLEMENTATION")
        if env_attn:
            return env_attn

        for impl in AttentionManager.ATTN_PRIORITIES.get(device, ["eager"]):
            for pkg, pkg_impl in AttentionManager.PACKAGE_TO_IMPL.items():
                if pkg_impl == impl and AttentionManager.check_package(pkg):
                    return impl
            if impl in ("eager", "sdpa"):
                return impl

        return "eager"

# 设备检测（CUDA/XPU 平级优先级，然后是 MPS -> CPU）
def detect_device():
    """检测可用设备"""
    try:
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            return "xpu", torch.bfloat16
        elif torch.backends.mps.is_available():
            return "mps", torch.float32
    except Exception as e:
        print(f"⚠ 设备检测异常: {e}")
    return "cpu", torch.float32

DEVICE, DTYPE = detect_device()
ATTN_IMPL = AttentionManager.select_attn(DEVICE)
AVAILABLE_ATTNS = AttentionManager.get_available_impls(DEVICE)

print(f"✓ 设备={DEVICE.upper()} dtype={DTYPE} attn={ATTN_IMPL}")
if len(AVAILABLE_ATTNS) > 1:
    print(f"✓ 可用: {', '.join(AVAILABLE_ATTNS.keys())}")
else:
    print(f"⚠ 仅可用eager实现")

# ============================================================================
# 进度跟踪器
# ============================================================================

class ProgressTracker:
    """支持控制台和WebUI双模式的进度跟踪"""

    def __init__(self, total: int, desc: str = "", web_mode: bool = False):
        self.total = total
        self.desc = desc
        self.current = 0
        self.web_mode = web_mode
        self.start_time = time.time()
        self.messages = []

        if not web_mode:
            self.pbar = tqdm(total=total, desc=desc, unit="块")
        else:
            self.pbar = None

    def update(self, n: int = 1, msg: str = ""):
        """更新进度"""
        self.current += n
        if msg:
            self.messages.append(f"[{self.current}/{self.total}] {msg}")

        if not self.web_mode:
            self.pbar.update(n)
            if msg:
                self.pbar.set_postfix_str(msg[:30])

    def get_progress_text(self) -> str:
        """获取进度文本（用于WebUI）"""
        elapsed = time.time() - self.start_time
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = (self.total - self.current) * avg_time
            speed = self.current / elapsed if elapsed > 0 else 0
        else:
            remaining = 0
            speed = 0

        progress_info = f"处理进度: {self.current}/{self.total} "
        if speed > 0:
            progress_info += f"({speed:.1f}块/秒, 剩余 {remaining:.0f}s)\n"
        else:
            progress_info += "\n"

        return progress_info + "\n".join(self.messages[-10:])  # 最后10条消息

    def close(self):
        """关闭进度条"""
        if self.pbar:
            self.pbar.close()

# 模型与处理器
MODEL_ID = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(MODEL_ID)

def load_model_with_fallback(model_id: str, attn_impl: str, device: str, dtype):
    """加载模型，支持Attention实现自动降级"""
    priorities = AttentionManager.ATTN_PRIORITIES.get(device, ["eager"])
    if attn_impl not in priorities:
        priorities.insert(0, attn_impl)

    for impl in priorities:
        try:
            print(f"  尝试: attn={impl}...", end=" ")
            m = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto", attn_implementation=impl
            )
            m.generation_config.max_length = None
            print("✓")
            return m, impl
        except Exception as e:
            print(f"✗ ({str(e)[:50]})")

    raise RuntimeError(f"无法加载模型: 所有Attention实现都失败")

model, actual_attn = load_model_with_fallback(MODEL_ID, ATTN_IMPL, DEVICE, DTYPE)
ATTN_IMPL = actual_attn

# torch.compile 优化（可选）
if os.environ.get("TORCH_COMPILE", "1").lower() in ("1", "true"):
    try:
        print("✓ 启用 torch.compile 优化")
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠ torch.compile 失败: {e}")

model.eval()
print(f"✓ 模型: 设备={DEVICE.upper()} attn={ATTN_IMPL}\n")

# ============================================================================
# 设备转移和缓存管理
# ============================================================================

def _to_device(batch):
    """将数据转移到设备，失败时降级到CPU"""
    try:
        if isinstance(batch, dict):
            return {k: v.to(device=DEVICE, dtype=DTYPE) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(b.to(device=DEVICE, dtype=DTYPE) if hasattr(b, 'to') else b 
                              for b in batch)
        else:
            return batch.to(device=DEVICE, dtype=DTYPE) if hasattr(batch, 'to') else batch
    except Exception:
        if DEVICE != "cpu":
            try:
                print(f"⚠ {DEVICE}转移失败，降级到CPU")
                if isinstance(batch, dict):
                    return {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in batch.items()}
                return batch.to("cpu") if hasattr(batch, 'to') else batch
            except:
                return batch
        return batch

def _clear_cache():
    """清理设备缓存"""
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "xpu":
            torch.xpu.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
    except:
        pass

def transcribe_chunk(audio_chunk, sr, time_offset=0.0):
    """转录音频块，支持失败重试"""
    batch = processor.apply_transcription_request(audio=audio_chunk, sampling_rate=sr)
    batch = _to_device(batch)

    for attempt in range(2):
        try:
            with torch.no_grad():
                outputs = model.generate(**batch, max_new_tokens=2048, repetition_penalty=1.1, do_sample=False)

            input_len = batch["input_ids"].shape[1]
            parsed = processor.decode(outputs[:, input_len:], return_format="parsed")[0]

            if isinstance(parsed, list):
                for seg in parsed:
                    if isinstance(seg, dict):
                        seg["Start"] = seg.get("Start", 0) + time_offset
                        seg["End"] = seg.get("End", 0) + time_offset
            return parsed
        except Exception as e:
            if attempt == 0:
                print(f"  ⚠ 重试: {str(e)[:50]}")
                _clear_cache()
            else:
                return None

def transcribe(audio_path, chunk_minutes: int):
    """分块转录音频，支持进度显示"""
    if not audio_path:
        return "未提供音频文件。", None

    try:
        # 检测是否在WebUI中运行
        web_mode = hasattr(sys, 'ps1') == False and 'gradio' in sys.modules

        sr_target = processor.feature_extractor.sampling_rate
        audio, sr = librosa.load(audio_path, sr=sr_target)
        total_dur = len(audio) / sr

        chunk_samples = max(1, int(chunk_minutes * 60 * sr))
        overlap = int(2 * sr)

        # 切分音频
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunks.append((audio[start:end], start / sr))
            if end == len(audio):
                break
            start += chunk_samples - overlap

        # 初始化进度跟踪
        progress = ProgressTracker(len(chunks), desc="转录", web_mode=web_mode)

        if not web_mode:
            print(f"音频: {total_dur:.1f}s, 块数: {len(chunks)}\n")

        segments = []
        for i, (chunk, offset) in enumerate(chunks, 1):
            try:
                parsed = transcribe_chunk(chunk, sr, offset)
                if parsed and isinstance(parsed, list):
                    segments.extend(parsed)
                    progress.update(1, f"✓ 块{i}成功")
                else:
                    progress.update(1, f"✗ 块{i}无效")
            except Exception as e:
                progress.update(1, f"✗ 块{i}失败")

            if web_mode and i % max(1, len(chunks) // 10) == 0:
                # 每处理10%时返回进度（WebUI）
                print(progress.get_progress_text(), flush=True)

            _clear_cache()

        progress.close()

        # 生成 SRT
        lines = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = seg.get("Content", "").strip()
            if not text or text == "[Silence]":
                continue
            lines.append(f"{len(lines) + 1}\n{format_srt_time(seg['Start'])} --> {format_srt_time(seg['End'])}\n{text}\n")

        srt_content = "\n".join(lines)
        with open("output.srt", "w", encoding="utf-8") as f:
            f.write(srt_content)

        if web_mode:
            print("✓ 转录完成!")

        return srt_content, "output.srt"

    except Exception as e:
        import traceback
        return f"错误: {e}\n{traceback.format_exc()}", None

def format_srt_time(seconds: float) -> str:
    """格式化SRT时间戳"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    hh, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

# Gradio UI
default_chunk = 20 if DEVICE == "mps" else 5

# 创建改进的接口，支持进度显示
with gr.Blocks(title="VVS SRT 生成") as iface:
    gr.Markdown(f"# VVS SRT 生成工具 [{DEVICE.upper()}]")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="上传音频")
            chunk_slider = gr.Slider(1, 60, value=default_chunk, step=1, label="分块(分钟)")
            submit_btn = gr.Button("开始转录", variant="primary")

        with gr.Column():
            progress_output = gr.Textbox(label="处理进度", lines=8, interactive=False)

    with gr.Row():
        srt_preview = gr.Textbox(lines=15, label="SRT 预览")
        srt_file = gr.File(label="下载 SRT")

    def transcribe_with_progress(audio_path, chunk_minutes):
        """包装transcribe函数以支持实时进度反馈"""
        # 重定向stdout以捕获进度信息
        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                srt_content, file_path = transcribe(audio_path, chunk_minutes)
            progress_text = captured_output.getvalue()
            if not progress_text:
                progress_text = "✓ 转录完成"
        except Exception as e:
            progress_text = f"错误: {e}"
            srt_content = f"转录失败: {e}"
            file_path = None

        return progress_text, srt_content, file_path

    submit_btn.click(
        fn=transcribe_with_progress,
        inputs=[audio_input, chunk_slider],
        outputs=[progress_output, srt_preview, srt_file]
    )

if __name__ == "__main__":
    iface.launch()
