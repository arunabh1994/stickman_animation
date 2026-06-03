# src/parallel_renderer.py
"""
Drop-in replacement for renderer.py that uses all CPU cores.
Each core renders a chunk of frames to a temp file.
ffmpeg concatenates the chunks into the final video.
"""
import multiprocessing as mp
import subprocess
import numpy as np
import os
import sys
import tempfile
import time
from tqdm import tqdm
import psutil

sys.path.insert(0, os.path.dirname(__file__))

from config    import VIDEO_WIDTH, VIDEO_HEIGHT, FPS, CHARACTER_CONFIG
from character import StickmanCharacter
from background import draw_scene
from timeline  import get_event_at_time, get_total_duration


# ─── Worker function (runs in separate process) ────────────────────────────────

def _render_chunk(args: tuple):
    """
    Render a range of frames and write them to a temporary .mp4 file.
    This function runs inside a worker process — no shared state with the main process.
    """
    chunk_id, start_frame, end_frame, timeline, script, temp_path, show_progress = args

    # Each worker builds its own character objects (deterministic because of the fixed seed)
    characters = {}
    for name, cfg in CHARACTER_CONFIG.items():
        characters[name] = StickmanCharacter(name, cfg["color"], cfg["slot"])

    n_frames = end_frame - start_frame

    # Start a local ffmpeg process — write raw frames, output to temp .mp4
    # Key difference from original: no audio here; chunks are video-only.
    # -movflags +faststart ensures the output is seekable (needed for concatenation).
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-pix_fmt", "bgr24",
        "-s",       f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r",       str(FPS),
        "-i",       "pipe:0",

        # "-c:v",     "libx264",
        # "-preset",  "fast",      # Balance speed vs file size
        # "-crf",     "21",
        "-c:v",     "h264_nvenc",
        "-preset",  "p4",          # p1=fastest, p7=best quality. p4 is balanced.
        "-rc",      "vbr",
        "-cq",      "21",          # Same quality as CRF 21 in libx264

        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temp_path,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin  = subprocess.PIPE,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )

    try:
        iterator = range(n_frames)
        if show_progress and chunk_id == 0:
            # Only chunk 0 shows a progress bar (avoid interleaved output)
            iterator = tqdm(iterator, desc=f"Rendering (chunk 0 of {chunk_id})",
                            position=0, leave=True)

        for local_frame in iterator:
            frame_num = start_frame + local_frame
            t = frame_num / FPS
            event = get_event_at_time(timeline, t)

            # Allocate frame
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

            # Background
            scene_id = (event["scene_id"] if event
                        else script["scenes"][0]["id"])
            draw_scene(frame, scene_id, t)

            # Characters
            for char_name, char in characters.items():
                is_speaking = (
                    event is not None
                    and event["type"] == "dialogue"
                    and event["character"] == char_name
                )
                if is_speaking:
                    t_rel   = t - event["start_time"]
                    phoneme = _get_phoneme(event["mouth_cues"], t_rel)
                    emotion = event["emotion"]
                    gesture = event["gesture"]
                else:
                    phoneme = "X"
                    emotion = "neutral"
                    gesture = "idle"

                char.draw(frame, t, phoneme, emotion, gesture, is_speaking)

            # Write to local ffmpeg
            proc.stdin.write(frame.tobytes())

    finally:
        proc.stdin.close()

    proc.wait()
    return chunk_id, temp_path


def _get_phoneme(mouth_cues: list, t_relative: float) -> str:
    for cue in mouth_cues:
        if cue["start"] <= t_relative < cue["end"]:
            return cue["value"]
    return "X"


# ─── Main parallel render function ────────────────────────────────────────────

def render_video_parallel(
    timeline:    list,
    script:      dict,
    audio_path:  str,
    output_path: str,
    n_workers:   int  = None,
    verbose:     bool = True,
):
    """
    Render video using all available CPU cores.

    Args:
        timeline:    Master event timeline
        script:      Parsed script dict
        audio_path:  Path to master mixed .wav audio
        output_path: Path for final .mp4 output
        n_workers:   Number of parallel processes. Defaults to cpu_count().
                     For 8-core CPU: 7 workers (leave 1 core for OS + ffmpeg)
        verbose:     Print progress
    """
    if n_workers is None:
        # Leave 1 core for OS + the ffmpeg processes themselves
        # n_workers = max(1, mp.cpu_count() - 1)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        max_by_ram   = max(1, int(available_gb / 0.35))   # ~350 MB per worker
        max_by_cpu   = max(1, mp.cpu_count() // 2)        # physical cores, not threads
        n_workers    = min(max_by_ram, max_by_cpu, 6)     # hard cap at 6 on 8 GB machines
        print(f"Auto-selected {n_workers} workers ({available_gb:.1f} GB RAM available)")


    total_duration = get_total_duration(timeline)
    total_frames   = int(total_duration * FPS)

    if verbose:
        print(f"Parallel render: {total_frames:,} frames, {n_workers} workers")
        print(f"Expected speedup: ~{n_workers * 0.85:.0f}× vs single-threaded")

    # ── Split frames into chunks ───────────────────────────────────────────────
    chunk_size = total_frames // n_workers
    chunks     = []

    os.makedirs("assets/temp_chunks", exist_ok=True)

    for i in range(n_workers):
        start = i * chunk_size
        end   = (i + 1) * chunk_size if i < n_workers - 1 else total_frames
        temp  = os.path.abspath(f"assets/temp_chunks/chunk_{i:03d}.mp4")
        chunks.append((i, start, end, timeline, script, temp, verbose))

    # ── Render chunks in parallel ──────────────────────────────────────────────
    t_start = time.time()

    # Windows requires freeze_support() and 'spawn' context
    ctx = mp.get_context("spawn")

    if verbose:
        print(f"\nDispatching {n_workers} workers...")

    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_render_chunk, chunks)

    elapsed = time.time() - t_start
    render_fps = total_frames / elapsed

    if verbose:
        print(f"\nAll chunks done in {elapsed:.1f}s  ({render_fps:.1f} frames/sec)")

    # ── Sort chunks by ID (pool.map preserves order, but be safe) ─────────────
    results.sort(key=lambda x: x[0])
    chunk_paths = [r[1] for r in results]

    # ── Concatenate chunks + add audio with ffmpeg ─────────────────────────────
    if verbose:
        print("Concatenating chunks and adding audio...")

    _concat_chunks(chunk_paths, audio_path, output_path)

    # ── Clean up temp files ────────────────────────────────────────────────────
    for path in chunk_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir("assets/temp_chunks")
    except OSError:
        pass  # Not empty — that's fine

    if verbose:
        total_time = time.time() - t_start
        print(f"\n✓ Video saved: {output_path}")
        print(f"  Total render time: {total_time:.0f}s  ({total_time/60:.1f} min)")


def _concat_chunks(chunk_paths: list, audio_path: str, output_path: str):
    """
    Use ffmpeg concat demuxer to join chunks losslessly, then mux audio.
    This avoids re-encoding — it is nearly instant (fast copy).
    """
    # Write the concat file list
    list_path = "assets/temp_chunks/concat_list.txt"
    os.makedirs(os.path.dirname(list_path), exist_ok=True)

    with open(list_path, "w") as f:
        for path in chunk_paths:
            # ffmpeg concat requires forward slashes even on Windows
            safe_path = path.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f",     "concat",
        "-safe",  "0",
        "-i",     list_path,       # Video chunks
        "-i",     audio_path,      # Master audio
        "-c:v",   "copy",          # No re-encode — just copy the H.264 stream
        "-c:a",   "aac",
        "-b:a",   "192k",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")

    # Clean up list file
    try:
        os.remove(list_path)
    except OSError:
        pass
