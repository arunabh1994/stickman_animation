import os
from pydub import AudioSegment
from config import AUDIO_DIR


def build_master_audio(timeline: list, output_path: str) -> str:
    """
    Concatenate all audio files according to the timeline.
    Fills gaps with silence so the audio track aligns with video frames.
    Returns path to the mixed .wav file.
    """
    total_duration_ms = int(get_total_duration_from_timeline(timeline) * 1000)
    master = AudioSegment.silent(duration=total_duration_ms)

    for event in timeline:
        if event["type"] != "dialogue":
            continue

        line_id    = event["line_id"]
        start_ms   = int(event["start_time"] * 1000)
        audio_path = os.path.join(AUDIO_DIR, f"{line_id}.wav")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        segment = AudioSegment.from_wav(audio_path)
        master  = master.overlay(segment, position=start_ms)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master.export(output_path, format="wav")
    print(f"✓ Master audio saved: {output_path}  ({total_duration_ms/1000:.1f}s)")
    return output_path


def get_total_duration_from_timeline(timeline: list) -> float:
    if not timeline:
        return 0.0
    return timeline[-1]["end_time"]
