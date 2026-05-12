import os

# 设置环境变量以启用更宽松的内存分配限制
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
import gradio as gr
import librosa
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

# =========================
# 设备设置
# =========================
# 检查可用设备
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.xpu.is_available():
    device = "xpu"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

# 根据设备选择注意力机制实现方式
if device == "cuda" or device == "xpu":
    attn_implementation = "flash_attention_2"
else:
    # MPS 或 CPU 不支持 flash_attention_2
    attn_implementation = "sdpa"

print(f"使用设备: {device}, 数据类型: {dtype}, 注意力机制: {attn_implementation}")

# 模型ID
model_id = "microsoft/VibeVoice-ASR-HF"
# 加载处理器和模型
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto"
)
model.eval()


# =========================
# SRT 时间格式化函数
# =========================
def format_srt_time(seconds):
    try:
        ms = int((seconds % 1) * 1000)
        seconds = int(seconds)
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    except:
        return "00:00:00,000"


# =========================
# 块转录处理函数
# =========================
def transcribe_chunk(audio_chunk, sr, time_offset=0.0):
    inputs = processor.apply_transcription_request(
        audio=audio_chunk,
        sampling_rate=sr,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            repetition_penalty=1.1,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]

    if isinstance(parsed, list):
        for seg in parsed:
            if isinstance(seg, dict):
                seg['Start'] = seg.get('Start', 0) + time_offset
                seg['End'] = seg.get('End', 0) + time_offset
    return parsed


# =========================
# 主转录处理函数
# =========================
def transcribe(audio_path, chunk_minutes):
    if audio_path is None:
        return "未提供音频文件。", None

    try:
        sr_target = processor.feature_extractor.sampling_rate
        audio, sr = librosa.load(audio_path, sr=sr_target)

        total_duration = len(audio) / sr
        chunk_samples = int(chunk_minutes * 60 * sr)
        overlap_samples = int(2 * sr)

        # 分割音频块
        chunks = []
        pos = 0
        while pos < len(audio):
            end = min(pos + chunk_samples, len(audio))
            chunks.append((audio[pos:end], pos / sr))
            if end == len(audio):
                break
            pos += chunk_samples - overlap_samples

        print(f"总共: {total_duration:.1f}秒 / {len(chunks)} 块")

        all_segments = []
        for i, (chunk, time_offset) in enumerate(chunks):
            print(f"处理中:  {i + 1}/{len(chunks)} 块 (偏移={time_offset:.1f}s)")
            parsed = transcribe_chunk(chunk, sr, time_offset)

            if isinstance(parsed, list):
                all_segments.extend(parsed)
            elif isinstance(parsed, str):
                print(f"块{i + 1} 解析失败: {parsed}")

            # 清理缓存
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "xpu":
                torch.xpu.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        # 生成 SRT 文件内容（忽略说话人、跳过静音）
        srt_content = ""
        seg_index = 1
        for seg in all_segments:
            if not isinstance(seg, dict):
                continue
            content = seg.get('Content', '').strip()
            if not content or content == '[Silence]':
                continue

            start_time = format_srt_time(seg.get('Start', 0))
            end_time = format_srt_time(seg.get('End', 0))
            srt_content += f"{seg_index}\n{start_time} --> {end_time}\n{content}\n\n"
            seg_index += 1

        output_filename = "output.srt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(srt_content)

        return srt_content, output_filename

    except Exception as e:
        import traceback
        return f"错误:\n{str(e)}\n\n{traceback.format_exc()}", None


# =========================
# Gradio 用户界面
# =========================
# 默认分割块大小（根据设备调整）
default_chunk = 20 if device == "mps" else 5

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath", label="上传音频文件"),
        gr.Slider(minimum=1, maximum=60, value=default_chunk,
                  step=1, label="分割块（分钟）"),
    ],
    outputs=[
        gr.Textbox(label="SRT 预览", lines=20),
        gr.File(label="下载 SRT 文件")
    ],
    title=f"VVS-字幕SRT生成（使用：{device.upper()}）"
)

if __name__ == "__main__":
    demo.launch()
