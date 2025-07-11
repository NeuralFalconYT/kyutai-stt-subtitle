#@title Works better without music and background noise
import itertools
import dataclasses
import julius
import sphn
import math
import json
import torch
import gc
import tqdm
import moshi.models


@dataclasses.dataclass
class TimestampedText:
    text: str
    timestamp: tuple[float, float]


def tokens_to_timestamped_text(text_tokens, tokenizer, frame_rate, end_of_padding_id, padding_token_id, offset_seconds):
    text_tokens = text_tokens.cpu().view(-1)
    sequence_timestamps = []

    def _tstmp(start, end):
        return (
            max(0, start / frame_rate - offset_seconds),
            max(0, end / frame_rate - offset_seconds),
        )

    def _decode(t):
        t = t[t > padding_token_id]
        return tokenizer.decode(t.numpy().tolist())

    def _decode_segment(start, end):
        start = int(start)
        end = int(end)
        text = _decode(text_tokens[start:end])
        words = text.split()
        if not words:
            return
        if len(words) == 1:
            sequence_timestamps.append(TimestampedText(text=text, timestamp=_tstmp(start, end)))
        else:
            for word in words[:-1]:
                n_tokens = len(tokenizer.encode(word))
                sequence_timestamps.append(TimestampedText(text=word, timestamp=_tstmp(start, start + n_tokens)))
                start += n_tokens
            sequence_timestamps.append(TimestampedText(text=words[-1], timestamp=_tstmp(start, end)))

    (segment_boundaries,) = torch.where(text_tokens == end_of_padding_id)
    if not segment_boundaries.numel():
        return []

    for i in range(len(segment_boundaries) - 1):
        _decode_segment(segment_boundaries[i] + 1, segment_boundaries[i + 1])

    last_start = int(segment_boundaries[-1] + 1)
    (last_end,) = torch.where(torch.isin(text_tokens[last_start:], torch.tensor([tokenizer.eos_id()])))
    last_end = last_start + (int(last_end[0]) if last_end.numel() else frame_rate)
    _decode_segment(last_start, last_end)

    return sequence_timestamps


def load_model(hf_repo="kyutai/stt-2.6b-en", device="cuda"):
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(hf_repo)
    mimi = info.get_mimi(device=device)
    moshi_model = info.get_moshi(device=device, dtype=torch.bfloat16)
    tokenizer = info.get_text_tokenizer()
    lm_gen = moshi.models.LMGen(moshi_model, temp=0, temp_text=0.0)

    return {
        "info": info,
        "mimi": mimi,
        "lm": moshi_model,
        "tokenizer": tokenizer,
        "lm_gen": lm_gen,
        "device": device
    }


def run_transcription(model, audio_path, save_json_path="word_timestamps.json"):
    info = model["info"]
    mimi = model["mimi"]
    lm_gen = model["lm_gen"]
    tokenizer = model["tokenizer"]
    device = model["device"]

    silence_prefix = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
    audio_delay = info.stt_config.get("audio_delay_seconds", 5.0)
    pad_id = info.raw_config.get("text_padding_token_id", 3)

    audio, sr = sphn.read(audio_path)
    audio = torch.from_numpy(audio).to(device)
    audio = julius.resample_frac(audio, sr, mimi.sample_rate)
    if audio.shape[-1] % mimi.frame_size != 0:
        pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
        audio = torch.nn.functional.pad(audio, (0, pad))

    prefix_chunks = math.ceil(silence_prefix * mimi.frame_rate)
    suffix_chunks = math.ceil(audio_delay * mimi.frame_rate)
    silence = torch.zeros((1, 1, mimi.frame_size), dtype=torch.float32, device=device)
    chunks = itertools.chain(
        itertools.repeat(silence, prefix_chunks),
        torch.split(audio[:, None], mimi.frame_size, dim=-1),
        itertools.repeat(silence, suffix_chunks)
    )

    all_tokens = []
    with mimi.streaming(1), lm_gen.streaming(1):
        for chunk in tqdm.tqdm(chunks):
            audio_tokens = mimi.encode(chunk)
            text_tokens = lm_gen.step(audio_tokens)
            if text_tokens is not None:
                all_tokens.append(text_tokens)

    utterance_tokens = torch.concat(all_tokens, dim=-1)
    offset = prefix_chunks / mimi.frame_rate + audio_delay
    timestamped = tokens_to_timestamped_text(
        utterance_tokens,
        tokenizer,
        mimi.frame_rate,
        end_of_padding_id=0,
        padding_token_id=pad_id,
        offset_seconds=offset,
    )

    transcription_words = []
    word_timestamps = []

    for t in timestamped:
        transcription_words.append(t.text)
        word_timestamps.append({
            "word": t.text,
            "start": t.timestamp[0],
            "end": t.timestamp[1]
        })

    transcription = " ".join(transcription_words)

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "transcription": transcription,
                "word_timestamps": word_timestamps
            }, f, indent=2)
        # print(f"Saved: {save_json_path}")

    return transcription, word_timestamps


def unload_model(model):
    model["mimi"].to("cpu")
    model["lm"].to("cpu")
    torch.cuda.empty_cache()
    del model
    gc.collect()
    print("Model unloaded and memory freed.")

import subprocess
import soundfile as sf
import os



def ensure_mono(audio_path: str) -> str:
    """
    Ensure an audio file is mono. If stereo, converts to mono using FFmpeg.
    Returns the path to the mono audio file (original or new).
    Raises RuntimeError if conversion fails.
    """
    try:
        with sf.SoundFile(audio_path) as f:
            channels = f.channels

        if channels == 1:
            print(f"✅ Audio is already mono: {audio_path}")
            return audio_path

        # print(f"⚠️ Audio is stereo (channels={channels}), converting to mono MP3...")

        base, _ = os.path.splitext(os.path.basename(audio_path))[0]
        mono_path = f"./subtitle/{base}_mono.mp3"

        # Remove stale output if exists
        if os.path.exists(mono_path):
            os.remove(mono_path)

        # Run FFmpeg conversion
        command = [
            "ffmpeg", "-i", audio_path,
            "-ac", "1",
            "-y", mono_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Validate conversion success
        if not os.path.isfile(mono_path) or os.path.getsize(mono_path) == 0:
            raise RuntimeError(f"FFmpeg failed: '{mono_path}' not created or is empty.")

        # print(f"✅ Converted to mono MP3: {mono_path}")
        return mono_path

    except Exception as e:
        raise RuntimeError(f"Mono conversion failed for '{audio_path}': {e}") from e

import os
import subprocess

def convert_video_to_mono_mp3(video_path):
    # Create output path by replacing extension
    base=os.path.splitext(os.path.basename(video_path))[0]
    mono_mp3_path = "./subtitle/"+base + "_mono.mp3"
    if os.path.exists(mono_mp3_path):
        os.remove(mono_mp3_path)
    # FFmpeg command
    command = [
        "ffmpeg", "-i", video_path,   # Input video file
        "-vn",                        # Remove video
        "-ac", "1",                   # Mono audio
        "-ar", "44100",               # Optional: Set sample rate
        "-y",                         # Overwrite output
        mono_mp3_path
    ]

    # Run command silently
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # print("✅ Saved to:", mono_mp3_path)
    return mono_mp3_path




import string

def write_word_srt(word_level_timestamps, output_file="word.srt", skip_punctuation=True):
    with open(output_file, "w", encoding="utf-8") as f:
        index = 1  # Track subtitle numbering separately

        for entry in word_level_timestamps:
            word = entry["word"]

            # Skip punctuation if enabled
            if skip_punctuation and all(char in string.punctuation for char in word):
                continue

            start_time = entry["start"]
            end_time = entry["end"]

            # Convert seconds to SRT time format (HH:MM:SS,mmm)
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                sec = int(seconds % 60)
                millisec = int((seconds % 1) * 1000)
                return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"

            start_srt = format_srt_time(start_time)
            end_srt = format_srt_time(end_time)

            # Write entry to SRT file
            f.write(f"{index}\n{start_srt} --> {end_srt}\n{word}\n\n")
            index += 1  # Increment subtitle number

def split_line_by_char_limit(text, max_chars_per_line=38):
    """
    Formats a text block into lines with a maximum character limit.
    Returns a list of strings (the lines).
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def write_sentence_srt(
    word_level_timestamps,
    output_file="subtitles_professional.srt",
    max_lines=2,
    max_duration_s=7.0,
    max_chars_per_line=38,
    hard_pause_threshold=0.5,
    merge_pause_threshold=0.4 # NEW: Threshold for merging single-word orphans
):
    """
    Creates professional-grade SRT files using a two-phase process:
    1. Generates visually-aware subtitles.
    2. Performs a post-processing pass to merge single-word orphans.
    """
    if not word_level_timestamps:
        return

    # --- PHASE 1: Generate good "draft" subtitles ---
    draft_subtitles = []
    i = 0
    while i < len(word_level_timestamps):
        start_time = word_level_timestamps[i]["start"]
        current_words = []

        j = i
        while j < len(word_level_timestamps):
            entry = word_level_timestamps[j]
            potential_words = current_words + [entry["word"]]
            potential_text = " ".join(potential_words)

            # Check hard limits before adding the word
            if len(split_line_by_char_limit(potential_text, max_chars_per_line)) > max_lines: break
            if (entry["end"] - start_time) > max_duration_s and current_words: break

            if j > i:
                prev_entry = word_level_timestamps[j-1]
                pause = entry["start"] - prev_entry["end"]
                if pause >= hard_pause_threshold: break
                if prev_entry["word"].endswith(('.','!','?')): break

            current_words.append(entry["word"])
            j += 1

        if not current_words:
            current_words.append(word_level_timestamps[i]["word"])
            j = i + 1

        text = " ".join(current_words)
        end_time = word_level_timestamps[j - 1]["end"]
        draft_subtitles.append({ "start": start_time, "end": end_time, "text": text })
        i = j

    # --- PHASE 2: Post-processing to merge single-word orphans ---
    if not draft_subtitles:
        return

    final_subtitles = [draft_subtitles[0]]
    for k in range(1, len(draft_subtitles)):
        prev_sub = final_subtitles[-1]
        current_sub = draft_subtitles[k]

        # Check if current subtitle is a single-word orphan
        is_single_word_orphan = len(current_sub["text"].split()) == 1
        pause_from_prev = current_sub["start"] - prev_sub["end"]

        if is_single_word_orphan and pause_from_prev < merge_pause_threshold:
            # Attempt to merge it backwards
            merged_text = prev_sub["text"] + " " + current_sub["text"]

            # Only merge if it doesn't violate the line limit
            if len(split_line_by_char_limit(merged_text, max_chars_per_line)) <= max_lines:
                # Success! Update the previous subtitle instead of adding a new one.
                prev_sub["text"] = merged_text
                prev_sub["end"] = current_sub["end"]
                continue # Skip adding the current_sub as it's been merged

        # If not merged, just add the current subtitle as is
        final_subtitles.append(current_sub)

    # --- Write the final, polished SRT file ---
    def format_srt_time(seconds):
        h, m, s, ms = int(seconds//3600), int((seconds%3600)//60), int(seconds%60), int((seconds%1)*1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sub_data in enumerate(final_subtitles, start=1):
            text = sub_data["text"].replace(" ,", ",").replace(" .", ".")
            formatted_lines = split_line_by_char_limit(text, max_chars_per_line)
            f.write(f"{idx}\n")
            f.write(f"{format_srt_time(sub_data['start'])} --> {format_srt_time(sub_data['end'])}\n")
            f.write("\n".join(formatted_lines) + "\n\n")

    # print(f"SRT file '{output_file}' created with orphan-merging logic.")


def srt_making(upload_file):
    root_path="./subtitle"
    os.makedirs(root_path, exist_ok=True)
    model = load_model(device="cuda")
    if upload_file.endswith(".mp4"):
      mono_audio_path = convert_video_to_mono_mp3(upload_file)
    else:
      mono_audio_path = ensure_mono(upload_file)
    base=os.path.splitext(os.path.basename(upload_file))[0]
    save_json_path=f"{root_path}/{base}_word.json"
    word_srt_path =f"{root_path}/{base}_word.srt"
    sentence_srt_path=f"{root_path}/{base}_sentence.srt"
    text_path=f"{root_path}/{base}_text.txt"
    transcription, word_level_timestamps = run_transcription(model, mono_audio_path,save_json_path)
    unload_model(model)
    write_word_srt(word_level_timestamps, output_file=word_srt_path, skip_punctuation=True)
    write_sentence_srt(
    word_level_timestamps,
    output_file=sentence_srt_path,
    max_lines=2,
    max_duration_s=7.0,
    max_chars_per_line=38,
    hard_pause_threshold=0.5,
    merge_pause_threshold=0.4
    )
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    return sentence_srt_path,word_srt_path,save_json_path,text_path,transcription




##################################
# audio_or_video_path = '/content/sample_fr_hibiki_crepes.mp3'  # @param {type: "string"}

# sentence_srt_path,word_srt_path,save_json_path,text_path,transcription=srt_making(audio_or_video_path)
# from IPython.display import clear_output
# clear_output()
# print(f"Sentence SRT path: {sentence_srt_path}")
# print(f"Word SRT path: {word_srt_path}")
# print(f"JSON path: {save_json_path}")
# print(f"Text path: {text_path}")
# print(f"Transcription: {transcription}")
import gradio as gr

def ui():
    with gr.Blocks() as demo:
        gr.Markdown("<center><h1 style='font-size: 40px;'>Kyutai STT Subtitle Generation</h1></center>")  
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(label='Upload Audio or Video File')
                with gr.Row():
                    generate_btn = gr.Button('🚀 Generate', variant='primary')
            with gr.Column():
                sentence_srt_path = gr.File(label='📜 Download Sentence-Level SRT')
                with gr.Accordion('Others', open=False):
                    word_srt_path = gr.File(label='Download Word-Level SRT')
                    save_json_path = gr.File(label='Download Raw Timestamp JSON')
                    text_path = gr.File(label='Download Transcription as Text File')
                    transcription = gr.Textbox(label='Transcription',lines=3)

        # Define inputs and outputs
        inputs = [file_upload]
        outputs = [sentence_srt_path, word_srt_path, save_json_path, text_path, transcription]

        # Only use the button to trigger the function
        generate_btn.click(srt_making, inputs=inputs, outputs=outputs)

        # Example input
        gr.Examples(examples=["./sample_fr_hibiki_crepes.mp3"], inputs=[file_upload])

    return demo
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
# def main(debug=True, share=True):
    demo = ui()
    demo.queue().launch(debug=debug, share=share)

if __name__ == "__main__":
    main()
