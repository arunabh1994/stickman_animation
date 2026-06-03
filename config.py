import os
from dotenv import load_dotenv

load_dotenv()

# ─── Video ────────────────────────────────────────────────────────────────────
VIDEO_WIDTH  = 1280
VIDEO_HEIGHT = 720
FPS          = 24
BACKGROUND_COLOR = (215, 205, 190)  # Warm cream (BGR for OpenCV)

# ─── Characters ───────────────────────────────────────────────────────────────
# Each character has a unique color (BGR) and a slot (0=far-left … 3=far-right)
CHARACTER_CONFIG = {
    "alice":   {"color": (50,  50,  210), "slot": 0},   # Red-ish
    "bob":     {"color": (200, 100,  50), "slot": 1},   # Blue-ish
    "charlie": {"color": (40,  160,  40), "slot": 2},   # Green
    "diana":   {"color": (180,  50, 180), "slot": 3},   # Purple
}

# Horizontal positions for each slot (fraction of video width)
SLOT_X = {0: 0.14, 1: 0.38, 2: 0.62, 3: 0.86}

# ─── Anatomy ──────────────────────────────────────────────────────────────────
HEAD_RADIUS    = 38
BODY_LENGTH    = 130
SHOULDER_WIDTH = 0      # 0 = single-point shoulder (classic stickman)
STICK_WIDTH    = 4      # Line thickness for body
GROUND_Y       = 605    # Y-coordinate where feet touch the floor

# ─── ElevenLabs ───────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_IDS = {
    "alice":   os.getenv("VOICE_ID_ALICE"),
    "bob":     os.getenv("VOICE_ID_BOB"),
    "charlie": os.getenv("VOICE_ID_CHARLIE"),
    "diana":   os.getenv("VOICE_ID_DIANA"),
}
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# ─── Rhubarb ──────────────────────────────────────────────────────────────────
import platform
RHUBARB_EXE = "./rhubarb/rhubarb.exe"# + (".exe" if platform.system() == "Windows" else "")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = "assets/scripts"
AUDIO_DIR   = "assets/audio"
LIPSYNC_DIR = "assets/lipsync"
OUTPUT_DIR  = "assets/output"

# ─── Timing ───────────────────────────────────────────────────────────────────
DEFAULT_PAUSE_BETWEEN_LINES = 0.65   # seconds of silence between dialogue lines
SCENE_TRANSITION_PAUSE      = 1.5    # seconds of silence between scenes