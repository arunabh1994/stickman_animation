# src/benchmark.py
"""
Benchmarks your render speed and projects full video duration.
Run: python src/benchmark.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import math
from config import VIDEO_WIDTH, VIDEO_HEIGHT, FPS, CHARACTER_CONFIG
from character import StickmanCharacter
from background import draw_scene

def run_benchmark(n_frames=200, verbose=True):
    print("=" * 55)
    print("  RENDER SPEED BENCHMARK")
    print("=" * 55)

    # Build characters (same as full render)
    chars = {}
    for name, cfg in CHARACTER_CONFIG.items():
        chars[name] = StickmanCharacter(name, cfg["color"], cfg["slot"])

    # Warm up (first few frames are always slower due to JIT-like effects)
    warmup = 10
    for i in range(warmup):
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        draw_scene(frame, "scene_001", i / FPS)
        for name, char in chars.items():
            char.draw(frame, i / FPS, "X", "neutral", "idle", False)

    # Timed benchmark
    print(f"\nRendering {n_frames} frames ({n_frames/FPS:.1f}s of content)...")
    start = time.perf_counter()

    for i in range(n_frames):
        t = i / FPS
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        draw_scene(frame, "scene_001", t)
        # One character speaking, three idle
        for j, (name, char) in enumerate(chars.items()):
            is_speaking = (j == 0)
            phoneme = "D" if is_speaking and i % 8 < 4 else "X"
            char.draw(frame, t, phoneme,
                      "happy" if is_speaking else "neutral",
                      "idle", is_speaking)

    elapsed = time.perf_counter() - start
    render_fps = n_frames / elapsed

    print(f"\nResults:")
    print(f"  Frames rendered:    {n_frames}")
    print(f"  Time elapsed:       {elapsed:.2f}s")
    print(f"  Render speed:       {render_fps:.2f} frames/second")

    print(f"\nProjected total render time:")
    for duration_min in [5, 10, 30, 45, 50]:
        total_frames = duration_min * 60 * FPS
        render_seconds = total_frames / render_fps
        render_hours = render_seconds / 3600
        if render_hours >= 1:
            print(f"  {duration_min:2d}-min video ({total_frames:,} frames): "
                  f"{render_hours:.1f} hours")
        else:
            print(f"  {duration_min:2d}-min video ({total_frames:,} frames): "
                  f"{render_seconds/60:.0f} minutes")

    print(f"\nBreakdown (estimated):")
    frame_time_ms = elapsed / n_frames * 1000
    print(f"  Time per frame:     {frame_time_ms:.1f} ms")
    if render_fps < 5:
        print(f"  Status:  SLOW — multiprocessing will help ~7x")
    elif render_fps < 20:
        print(f"  Status:  MODERATE — multiprocessing will help ~5x")
    else:
        print(f"  Status:  GOOD")

    return render_fps

if __name__ == "__main__":
    run_benchmark()
