from config import DEFAULT_PAUSE_BETWEEN_LINES, SCENE_TRANSITION_PAUSE


def build_timeline(script: dict, audio_map: dict, lipsync_map: dict) -> list:
    """
    Build the master rendering timeline.

    Returns a list of events sorted by start_time. Each event is a dict:

    Dialogue event:
    {
        "type":         "dialogue",
        "line_id":      str,
        "character":    str,
        "scene_id":     str,
        "emotion":      str,
        "gesture":      str,
        "start_time":   float,   ← absolute seconds from video start
        "end_time":     float,
        "duration":     float,
        "mouth_cues":   list     ← Rhubarb phoneme list
    }

    Pause event:
    {
        "type":       "pause",
        "start_time": float,
        "end_time":   float,
        "scene_id":   str,
        "characters": list       ← which characters are on screen
    }
    """
    timeline = []
    cursor   = 0.0   # current time pointer (seconds)
    prev_scene_id = None

    for scene in script["scenes"]:
        scene_id = scene["id"]
        characters_on_screen = list(script["characters"].keys())  # All 4 always visible

        # Gap between scenes
        if prev_scene_id is not None:
            timeline.append({
                "type":       "pause",
                "start_time": cursor,
                "end_time":   cursor + SCENE_TRANSITION_PAUSE,
                "scene_id":   scene_id,
                "characters": characters_on_screen,
            })
            cursor += SCENE_TRANSITION_PAUSE

        for line in scene["dialogue"]:
            lid      = line["id"]
            duration = audio_map[lid]["duration"]
            pause    = float(line.get("pause_after", DEFAULT_PAUSE_BETWEEN_LINES))

            # Dialogue event
            timeline.append({
                "type":       "dialogue",
                "line_id":    lid,
                "character":  line["character"],
                "scene_id":   scene_id,
                "emotion":    line["emotion"],
                "gesture":    line["gesture"],
                "start_time": cursor,
                "end_time":   cursor + duration,
                "duration":   duration,
                "mouth_cues": lipsync_map[lid],
            })
            cursor += duration

            # Pause after line
            if pause > 0:
                timeline.append({
                    "type":       "pause",
                    "start_time": cursor,
                    "end_time":   cursor + pause,
                    "scene_id":   scene_id,
                    "characters": characters_on_screen,
                })
                cursor += pause

        prev_scene_id = scene_id

    return timeline


def get_event_at_time(timeline: list, t: float) -> dict:
    """
    Binary-search the timeline to find which event covers time t.
    Returns the event dict, or None if t is beyond the timeline.
    """
    for event in timeline:
        if event["start_time"] <= t < event["end_time"]:
            return event
    return None


def get_total_duration(timeline: list) -> float:
    if not timeline:
        return 0.0
    return timeline[-1]["end_time"]
