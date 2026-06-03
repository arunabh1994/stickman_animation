import subprocess
import numpy as np
import cv2
import math
from tqdm import tqdm
from typing import List, Dict

from config import (
    VIDEO_WIDTH, VIDEO_HEIGHT, FPS, CHARACTER_CONFIG, SLOT_X,
    GROUND_Y, BACKGROUND_COLOR
)
from character import StickmanCharacter
from background import draw_scene
from timeline import get_event_at_time, get_total_duration


def build_characters(script: dict) -> Dict[str, StickmanCharacter]:
    """Create a StickmanCharacter for each character defined in the script."""
    chars = {}
    for name, cfg in CHARACTER_CONFIG.items():
        chars[name] = StickmanCharacter(
            name  = name,
            color = cfg["color"],
            slot  = cfg["slot"],
        )
    return chars


def render_video(
    timeline: List[dict],
    script:   dict,
    audio_path: str,
    output_path: str,
    verbose: bool = True,
):
    """
    Render all frames and pipe them directly to ffmpeg for encoding.
    No frames are saved to disk.

    Args:
        timeline:    Master event timeline from timeline.py
        script:      Parsed script dict (for scene info)
        audio_path:  Path to master mixed audio .wav
        output_path: Path for the output .mp4 file
    """
    total_duration = get_total_duration(timeline)
    total_frames   = int(total_duration * FPS)
    characters     = build_characters(script)

    print(f"Rendering {total_frames:,} frames  ({total_duration:.1f}s @ {FPS}fps)")
    print(f"Output: {output_path}")

    # ── Start ffmpeg process ──────────────────────────────────────────────────
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-pix_fmt", "bgr24",
        "-s",       f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r",       str(FPS),
        "-i",       "pipe:0",      # Video from stdin
        "-i",       audio_path,    # Audio from file
        "-c:v",     "libx264",
        "-preset",  "fast",
        "-crf",     "21",
        "-pix_fmt", "yuv420p",     # Required for wide compatibility
        "-c:a",     "aac",
        "-b:a",     "192k",
        "-shortest",               # Stop when the shorter stream ends
        output_path,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin  = subprocess.PIPE,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )

    # ── Frame rendering loop ──────────────────────────────────────────────────
    try:
        iterator = tqdm(range(total_frames), desc="Rendering frames") if verbose else range(total_frames)

        for frame_num in iterator:
            t     = frame_num / FPS
            event = get_event_at_time(timeline, t)

            # Allocate frame
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

            # Background
            scene_id = event["scene_id"] if event else script["scenes"][0]["id"]
            draw_scene(frame, scene_id, t)

            # Characters
            for char_name, char in characters.items():
                is_speaking = (
                    event is not None and
                    event["type"] == "dialogue" and
                    event["character"] == char_name
                )

                if is_speaking:
                    t_relative = t - event["start_time"]
                    phoneme    = _get_phoneme(event["mouth_cues"], t_relative)
                    emotion    = event["emotion"]
                    gesture    = event["gesture"]
                else:
                    phoneme = "X"      # Mouth closed
                    emotion = "neutral"
                    gesture = "idle"

                char.draw(frame, t, phoneme, emotion, gesture, is_speaking)

            # Pipe raw BGR bytes to ffmpeg
            proc.stdin.write(frame.tobytes())

    except BrokenPipeError:
        # ffmpeg crashed — grab its error output
        _, err = proc.communicate()
        raise RuntimeError(f"ffmpeg pipe broke:\n{err.decode()}")

    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()

    proc.wait()

    if proc.returncode != 0:
        _, err = proc.stderr.read(), None
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    print(f"\n✓ Video saved: {output_path}")


def _get_phoneme(mouth_cues: list, t_relative: float) -> str:
    """Return the Rhubarb phoneme active at t_relative seconds into the line."""
    for cue in mouth_cues:
        if cue["start"] <= t_relative < cue["end"]:
            return cue["value"]
    return "X"
