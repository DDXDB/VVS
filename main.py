import gradio as gr
import librosa
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

# =========================
# 设备设置
# =========================
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

if device == "cuda" or device == "xpu":

    attn_implementation = "flash_attention_2"

else:
    # MPS  / CPU don't support flash_attention_2
    attn_implementation = "sdpa"

print(f"Using device: {device}, dtype: {dtype}, attn_implementation: {attn_implementation}")

model_id = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(model_id)
attn_implementation = attn_implementation
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto"
)
model.eval()


# =========================
# SRT格式
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
# 块处理
# =========================
def transcribe_chunk(audio_chunk, sr, time_offset=0.0):
    inputs = processor.apply_transcription_request(
        audio=audio_chunk,
        sampling_rate=sr,
    ).to(device=device, dtype=dtype)  # ← 修正箇所

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
# 主处理
# =========================
def transcribe(audio_path, chunk_minutes):
    if audio_path is None:
        return "No audio file provided.", None

    try:
        sr_target = processor.feature_extractor.sampling_rate
        audio, sr = librosa.load(audio_path, sr=sr_target)

        total_duration = len(audio) / sr
        chunk_samples = int(chunk_minutes * 60 * sr)
        overlap_samples = int(2 * sr)

        # チャンク分割
        chunks = []
        pos = 0
        while pos < len(audio):
            end = min(pos + chunk_samples, len(audio))
            chunks.append((audio[pos:end], pos / sr))
            if end == len(audio):
                break
            pos += chunk_samples - overlap_samples

        print(f"总: {total_duration:.1f}秒 / 块: {len(chunks)}")

        all_segments = []
        for i, (chunk, time_offset) in enumerate(chunks):
            print(f"处理: 块 {i + 1}/{len(chunks)} (offset={time_offset:.1f}s)")
            parsed = transcribe_chunk(chunk, sr, time_offset)

            if isinstance(parsed, list):
                all_segments.extend(parsed)
            elif isinstance(parsed, str):
                print(f"块{i + 1} 解析失败: {parsed}")

            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "xpu":
                torch.xpu.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        # SRT生成（話者なし・無音スキップ）
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
# Gradio UI
# =========================
default_chunk = 20 if device == "mps" else 5

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Slider(minimum=1, maximum=60, value=default_chunk,
                  step=1, label="分割块（分）"),
    ],
    outputs=[
        gr.Textbox(label="SRT Preview", lines=20),
        gr.File(label="Download SRT File")
    ],
    title=f"VVS-字幕SRT生成（{device.upper()}）"
)

if __name__ == "__main__":
    demo.launch()
