import json
import os
from typing import List, Dict, Any

VALID_EMOTIONS  = {"neutral","happy","laughing","sad","angry","surprised","serious","excited","sarcastic","thinking"}
VALID_GESTURES  = {"idle","arms_up","arms_crossed","wave_right","wave_left","point_right","point_left","thinking","shrug"}

def load_script(script_path: str) -> Dict[str, Any]:
    """
    Load and validate a script JSON file.
    Returns the parsed script dict or raises ValueError with a clear message.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    _validate(script)
    return script


def _validate(script: Dict):
    required_top = {"title", "characters", "scenes"}
    missing = required_top - set(script.keys())
    if missing:
        raise ValueError(f"Script is missing top-level keys: {missing}")

    char_names = set(script["characters"].keys())

    for s_idx, scene in enumerate(script["scenes"]):
        if "id" not in scene or "dialogue" not in scene:
            raise ValueError(f"Scene {s_idx} is missing 'id' or 'dialogue'")

        for d_idx, line in enumerate(scene["dialogue"]):
            loc = f"Scene '{scene['id']}', line {d_idx}"
            if "character" not in line or "text" not in line:
                raise ValueError(f"{loc}: missing 'character' or 'text'")
            if line["character"] not in char_names:
                raise ValueError(f"{loc}: unknown character '{line['character']}'")
            if "emotion" in line and line["emotion"] not in VALID_EMOTIONS:
                raise ValueError(f"{loc}: unknown emotion '{line['emotion']}'")
            if "gesture" in line and line["gesture"] not in VALID_GESTURES:
                raise ValueError(f"{loc}: unknown gesture '{line['gesture']}'")

            # Fill defaults
            line.setdefault("emotion", "neutral")
            line.setdefault("gesture", "idle")
            line.setdefault("pause_after", 0.65)
            line.setdefault("id", f"{scene['id']}_line{d_idx:03d}")


def get_all_lines(script: Dict) -> List[Dict]:
    """Flatten all dialogue lines across all scenes into a single list."""
    lines = []
    for scene in script["scenes"]:
        for line in scene["dialogue"]:
            lines.append({**line, "scene_id": scene["id"]})
    return lines
