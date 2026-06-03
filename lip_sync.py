import os
import json
import subprocess
from config import RHUBARB_EXE, LIPSYNC_DIR, AUDIO_DIR


def run_rhubarb(line: dict, audio_path: str) -> str:
    """
    Run Rhubarb Lip Sync on a .wav file.
    Produces a JSON file with phoneme timings.
    Returns path to the JSON output file.
    Skips if JSON already exists (caching).
    """
    os.makedirs(LIPSYNC_DIR, exist_ok=True)
    json_path = os.path.join(LIPSYNC_DIR, f"{line['id']}.json")

    if os.path.exists(json_path):
        return json_path   # Cache hit

    # Write a dialog file — providing the transcript makes Rhubarb MORE accurate
    dialog_path = json_path.replace(".json", ".txt")
    with open(dialog_path, "w", encoding="utf-8") as f:
        f.write(line["text"])

    cmd = [
        RHUBARB_EXE,
        "-r", "phonetic",          # Use phonetic analysis (fast, good quality)
        "-f", "json",              # Output format
        "--dialogFile", dialog_path,
        "-o", json_path,
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Rhubarb failed for {line['id']}:\n{result.stderr}")

    return json_path


def load_lipsync_data(json_path: str) -> list:
    """
    Load Rhubarb JSON and return list of mouth cue dicts.
    Each item: {"start": float, "end": float, "value": str}
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("mouthCues", [])


def get_phoneme_at_time(mouth_cues: list, t_relative: float) -> str:
    """
    Given elapsed time within a dialogue line, return the current Rhubarb phoneme.
    t_relative = seconds since this line started playing.
    """
    for cue in mouth_cues:
        if cue["start"] <= t_relative < cue["end"]:
            return cue["value"]
    return "X"   # Default: closed mouth


def run_all_lipsync(lines: list, audio_map: dict, verbose: bool = True) -> dict:
    """
    Run Rhubarb for every line that has audio.
    Returns {line_id: [mouth_cues]}
    """
    from tqdm import tqdm
    results = {}
    iterator = tqdm(lines, desc="Running lip sync") if verbose else lines

    for line in iterator:
        lid       = line["id"]
        audio_path = audio_map[lid]["path"]
        json_path  = run_rhubarb(line, audio_path)
        results[lid] = load_lipsync_data(json_path)

    return results
