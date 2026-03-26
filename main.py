import os
import time
import traceback
from pathlib import Path
from typing import List, Dict

import torch
import gradio as gr
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

# === 配置 ===
MODEL_ID = "microsoft/VibeVoice-ASR-HF"
ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "model"

# 创建目录
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 初始化模型
if torch.xpu.is_available():
    device = "xpu"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"使用设备: {device}")

# 加载模型和处理器
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    cache_dir=MODEL_DIR,
    torch_dtype=torch.float16 if device not in ["cpu", "mps"] else torch.float32
)
print(f"模型加载完成，设备: {model.device}")


def extract_audio_from_video(video_path: str, output_audio_path: str):
    """使用ffmpeg从视频中提取音频"""
    import subprocess
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path,
        "-y"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e}")
        return False


def generate_srt(transcription_data: List[Dict]) -> str:
    """生成SRT格式的字幕文件"""
    srt_content = []
    for i, segment in enumerate(transcription_data, 1):
        start_time = segment["Start"]
        end_time = segment["End"]
        content = segment["Content"]

        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millisecs = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

        start_str = format_time(start_time)
        end_str = format_time(end_time)

        srt_content.append(str(i))
        srt_content.append(f"{start_str} --> {end_str}")
        srt_content.append(content)
        srt_content.append("")

    return "\n".join(srt_content)


def process_video(video_file, prompt):
    try:
        # 保存上传的视频文件
        video_path = CACHE_DIR / f"input_{int(time.time())}.mp4"
        with open(video_file.name, "rb") as src:
            with open(video_path, "wb") as dst:
                dst.write(src.read())

        # 提取音频
        audio_path = CACHE_DIR / f"audio_{int(time.time())}.wav"
        if not extract_audio_from_video(str(video_path), str(audio_path)):
            raise Exception("无法从视频中提取音频")

        # 转换为模型输入
        inputs = processor.apply_transcription_request(
            audio=str(audio_path),
            prompt=prompt,
        ).to(model.device, model.dtype)

        # 生成字幕
        with torch.no_grad():
            output_ids = model.generate(**inputs)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            transcription = processor.decode(generated_ids, return_format="parsed")[0]

        # 生成SRT文件
        srt_content = generate_srt(transcription)
        srt_filename = f"subtitle_{int(time.time())}.srt"
        srt_path = OUTPUT_DIR / srt_filename
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # 清理临时文件
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except Exception as e:
            print(f"清理临时文件时出错: {e}")

        return srt_path

    except Exception as e:
        print(traceback.format_exc())
        raise gr.Error(str(e))


# Gradio UI
with gr.Blocks(title="视频字幕生成器") as demo:
    gr.Markdown("# 🎬 视频字幕生成器")
    gr.Markdown("上传视频文件，生成 SRT 字幕文件。")

    with gr.Row():
        video_input = gr.File(label="上传视频文件（支持 mp4）")
        prompt_input = gr.Textbox(label="提示词（可选）", placeholder="例如：中文语境下的自然语言")

    generate_btn = gr.Button("生成字幕")

    with gr.Row():
        srt_output = gr.File(label="生成的 SRT 字幕文件")

    generate_btn.click(
        fn=process_video,
        inputs=[video_input, prompt_input],
        outputs=[srt_output]
    )

if __name__ == "__main__":
    demo.launch()
