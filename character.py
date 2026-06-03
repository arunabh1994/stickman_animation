import cv2
import numpy as np
import math
import random
from config import (
    HEAD_RADIUS, STICK_WIDTH, GROUND_Y,
    VIDEO_WIDTH, VIDEO_HEIGHT, SLOT_X
)
import hashlib

class StickmanCharacter:
    """
    A mathematically-defined stickman with full animation support.
    Drawn every frame using OpenCV primitives (circles, lines, ellipses).
    """

    # def __init__(self, name: str, color: tuple, slot: int):
    #     self.name    = name
    #     self.color   = color                               # BGR tuple
    #     self.cx      = int(SLOT_X[slot] * VIDEO_WIDTH)    # Center X on screen
    #     self.ground  = GROUND_Y

    #     # Blink rhythm: each character has a slightly different blink rate
    #     self._blink_period   = 3.8 + random.uniform(-0.5, 0.5)  # seconds between blinks
    #     self._blink_duration = 0.12                               # seconds the eye stays closed

    def __init__(self, name: str, color: tuple, slot: int):
        self.name   = name
        self.color  = color
        self.cx     = int(SLOT_X[slot] * VIDEO_WIDTH)
        self.ground = GROUND_Y

        # CRITICAL: Use deterministic seed based on character name.
        # This guarantees identical blink timing across all parallel worker processes.
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2 ** 31)
        rng  = __import__("random").Random(seed)
        self._blink_period   = 3.8 + rng.uniform(-0.5, 0.5)
        self._blink_duration = 0.12

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, t: float,
             phoneme: str, emotion: str, gesture: str,
             is_speaking: bool):
        """
        Draw this character onto `frame` for time `t`.

        Args:
            frame:      OpenCV image array (H, W, 3) BGR
            t:          Absolute video time in seconds
            phoneme:    Rhubarb mouth code ('X','A'-'H') — 'X' when idle
            emotion:    Emotion string
            gesture:    Gesture string
            is_speaking: True only for the character currently delivering dialogue
        """
        joints  = self._compute_joints(t, emotion, gesture)
        hc      = joints["head_center"]
        aa      = cv2.LINE_AA
        tk      = STICK_WIDTH

        # 1. Ground shadow
        cv2.ellipse(frame, (self.cx, self.ground + 6),
                    (46, 9), 0, 0, 360, (175, 165, 155), -1)

        # 2. Legs
        cv2.line(frame, joints["hip"], joints["left_leg"],  self.color, tk, aa)
        cv2.line(frame, joints["hip"], joints["right_leg"], self.color, tk, aa)

        # 3. Feet (short horizontal lines)
        cv2.line(frame, joints["left_leg"],  joints["left_foot"],  self.color, tk, aa)
        cv2.line(frame, joints["right_leg"], joints["right_foot"], self.color, tk, aa)

        # 4. Body
        cv2.line(frame, joints["hip"], joints["shoulder"], self.color, tk, aa)

        # 5. Arms
        cv2.line(frame, joints["shoulder"], joints["left_arm"],  self.color, tk, aa)
        cv2.line(frame, joints["shoulder"], joints["right_arm"], self.color, tk, aa)

        # 6. Neck
        neck_bot = (hc[0], hc[1] + HEAD_RADIUS)
        cv2.line(frame, joints["shoulder"], neck_bot, self.color, tk - 1, aa)

        # 7. Head circle
        cv2.circle(frame, hc, HEAD_RADIUS, self.color, tk, aa)

        # 8. Eyes
        is_blinking = self._is_blinking(t)
        self._draw_eyes(frame, hc, emotion, is_blinking)

        # 9. Mouth
        self._draw_mouth(frame, hc, phoneme, emotion, is_speaking)

        # 10. Name label
        self._draw_label(frame, hc)

        # 11. Speaking indicator (bouncing dots)
        if is_speaking and phoneme not in ("X", "A"):
            self._draw_speaking_dots(frame, hc, t)

    # ─────────────────────────────────────────────────────────────────────────
    # Joint Computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_joints(self, t: float, emotion: str, gesture: str) -> dict:
        cx  = self.cx
        gnd = self.ground

        # Breathing: gentle vertical oscillation
        breathe = 2.5 * math.sin(t * 2 * math.pi * 0.22)

        # Emotion-specific body modifications
        jitter = 0
        lean   = 0
        bounce = 0

        if emotion == "laughing":
            jitter  = int(4 * math.sin(t * 12))
            bounce  = abs(5 * math.sin(t * 7))
        elif emotion == "excited":
            bounce  = abs(4 * math.sin(t * 6))
        elif emotion == "surprised":
            lean    = -8   # Lean back slightly

        # Core joint positions (bottom-up)
        hip_y      = gnd - 110
        shoulder_y = int(hip_y - 125 + breathe + bounce)
        head_y     = shoulder_y - 48

        shoulder   = (cx + jitter, shoulder_y)
        hip        = (cx, hip_y)

        # Arm endpoints (gesture-driven)
        la, ra = self._arm_endpoints(cx, shoulder_y, hip_y, gesture, emotion, t, jitter)

        return {
            "head_center": (cx + jitter + lean, head_y),
            "shoulder":    shoulder,
            "hip":         hip,
            "left_arm":    la,
            "right_arm":   ra,
            "left_leg":    (cx - 35, gnd),
            "right_leg":   (cx + 35, gnd),
            "left_foot":   (cx - 58, gnd),
            "right_foot":  (cx + 58, gnd),
        }

    def _arm_endpoints(self, cx, shoulder_y, hip_y, gesture, emotion, t, jitter):
        """Return (left_arm_end, right_arm_end) based on gesture and emotion."""
        # Defaults: arms hanging naturally
        la = (cx - 68 + jitter, hip_y - 8)
        ra = (cx + 68 + jitter, hip_y - 8)

        if gesture == "arms_up" or emotion == "excited":
            la = (cx - 62, shoulder_y - 55)
            ra = (cx + 62, shoulder_y - 55)

        elif gesture == "arms_crossed":
            la = (cx + 32, hip_y + 12)
            ra = (cx - 32, hip_y + 12)

        elif gesture == "wave_right":
            wave = int(18 * math.sin(t * 6))
            la   = (cx - 68, hip_y - 8)
            ra   = (cx + 55, shoulder_y - 55 + wave)

        elif gesture == "wave_left":
            wave = int(18 * math.sin(t * 6))
            la   = (cx - 55, shoulder_y - 55 + wave)
            ra   = (cx + 68, hip_y - 8)

        elif gesture == "point_right":
            la = (cx - 68, hip_y - 8)
            ra = (cx + 100, shoulder_y - 10)

        elif gesture == "point_left":
            la = (cx - 100, shoulder_y - 10)
            ra = (cx + 68, hip_y - 8)

        elif gesture == "thinking":
            la = (cx - 68, hip_y - 8)
            ra = (cx + 22, shoulder_y + 18)   # Hand near chin

        elif gesture == "shrug":
            la = (cx - 62, shoulder_y - 22)
            ra = (cx + 62, shoulder_y - 22)

        elif emotion == "laughing":
            jit = int(6 * math.sin(t * 9))
            la  = (cx - 82 + jit, hip_y - 2)
            ra  = (cx + 82 - jit, hip_y - 2)

        return (
            (int(la[0]), int(la[1])),
            (int(ra[0]), int(ra[1]))
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Face Drawing
    # ─────────────────────────────────────────────────────────────────────────

    def _is_blinking(self, t: float) -> bool:
        phase = t % self._blink_period
        return phase > (self._blink_period - self._blink_duration)

    def _draw_eyes(self, frame, hc, emotion, is_blinking):
        cx, cy   = hc
        eye_y    = cy - 14
        lx       = cx - 14
        rx       = cx + 14
        color    = self.color
        aa       = cv2.LINE_AA

        if is_blinking or emotion == "sleeping":
            # Closed: thin horizontal lines
            cv2.line(frame, (lx - 7, eye_y), (lx + 7, eye_y), color, 2, aa)
            cv2.line(frame, (rx - 7, eye_y), (rx + 7, eye_y), color, 2, aa)

        elif emotion == "laughing":
            # Squinted arcs (curved upward = closed-happy)
            cv2.ellipse(frame, (lx, eye_y + 2), (7, 4), 0, 190, 350, color, 2, aa)
            cv2.ellipse(frame, (rx, eye_y + 2), (7, 4), 0, 190, 350, color, 2, aa)

        elif emotion == "surprised":
            # Big wide-open circles
            cv2.circle(frame, (lx, eye_y), 8, color, 2, aa)
            cv2.circle(frame, (rx, eye_y), 8, color, 2, aa)
            cv2.circle(frame, (lx, eye_y), 3, color, -1, aa)   # Pupils
            cv2.circle(frame, (rx, eye_y), 3, color, -1, aa)

        elif emotion == "angry":
            # Angled eyebrows + small eyes
            cv2.line(frame, (lx - 8, eye_y - 9), (lx + 6, eye_y - 4), color, 2, aa)
            cv2.line(frame, (rx + 8, eye_y - 9), (rx - 6, eye_y - 4), color, 2, aa)
            cv2.circle(frame, (lx, eye_y + 1), 5, color, -1, aa)
            cv2.circle(frame, (rx, eye_y + 1), 5, color, -1, aa)

        elif emotion in ("happy", "excited"):
            # Normal eyes + raised eyebrows
            cv2.line(frame, (lx - 7, eye_y - 10), (lx + 7, eye_y - 7), color, 2, aa)
            cv2.line(frame, (rx - 7, eye_y - 7),  (rx + 7, eye_y - 10), color, 2, aa)
            cv2.circle(frame, (lx, eye_y), 5, color, -1, aa)
            cv2.circle(frame, (rx, eye_y), 5, color, -1, aa)

        elif emotion in ("sad", "thinking"):
            # Slightly drooping eyes
            cv2.line(frame, (lx - 7, eye_y - 4), (lx + 7, eye_y - 8), color, 2, aa)
            cv2.line(frame, (rx - 7, eye_y - 8), (rx + 7, eye_y - 4), color, 2, aa)
            cv2.circle(frame, (lx, eye_y + 1), 5, color, -1, aa)
            cv2.circle(frame, (rx, eye_y + 1), 5, color, -1, aa)

        else:
            # Neutral: solid dots
            cv2.circle(frame, (lx, eye_y), 5, color, -1, aa)
            cv2.circle(frame, (rx, eye_y), 5, color, -1, aa)

    def _draw_mouth(self, frame, hc, phoneme, emotion, is_speaking):
        """
        If is_speaking: draw the Rhubarb phoneme mouth shape.
        If idle:        draw the emotion resting mouth.
        """
        cx, cy    = hc
        mouth_cx  = cx
        mouth_cy  = cy + 18
        color     = self.color
        aa        = cv2.LINE_AA
        dark      = (25, 18, 18)   # Interior of open mouth

        if is_speaking:
            # ── Phoneme-driven shapes ─────────────────────────────────────
            shapes = {
                "X": ("line",    0,   0),    # Silence
                "A": ("line",    0,   0),    # Rest / lips together
                "B": ("ellipse", 14,  4),    # M/B/P — barely open
                "C": ("ellipse", 16,  8),    # Open vowel
                "D": ("ellipse", 14, 13),    # TH — wider
                "E": ("ellipse", 20, 16),    # EE — very wide
                "F": ("teeth",   16,  8),    # F/V — upper teeth visible
                "G": ("ellipse", 14, 14),    # G/K — back open
                "H": ("circle",   9,  0),    # CH/SH/W — rounded/puckered
            }
            shape, w, h = shapes.get(phoneme, shapes["X"])

            if shape == "line":
                cv2.line(frame, (mouth_cx - 13, mouth_cy),
                                (mouth_cx + 13, mouth_cy), color, 2, aa)

            elif shape == "ellipse":
                cv2.ellipse(frame, (mouth_cx, mouth_cy), (w, h),
                            0, 0, 360, color, 2, aa)
                cv2.ellipse(frame, (mouth_cx, mouth_cy), (w - 2, h - 2),
                            0, 0, 360, dark, -1)

            elif shape == "circle":
                cv2.circle(frame, (mouth_cx, mouth_cy), w, color, 2, aa)
                cv2.circle(frame, (mouth_cx, mouth_cy), w - 2, dark, -1)

            elif shape == "teeth":
                cv2.ellipse(frame, (mouth_cx, mouth_cy), (w, h),
                            0, 0, 360, color, 2, aa)
                cv2.ellipse(frame, (mouth_cx, mouth_cy), (w - 2, h - 2),
                            0, 0, 360, (240, 240, 240), -1)   # White teeth
                cv2.line(frame, (mouth_cx - w + 3, mouth_cy),
                                (mouth_cx + w - 3, mouth_cy), color, 1)

        else:
            # ── Emotion resting mouth ──────────────────────────────────────
            if emotion in ("happy", "excited"):
                # Smile arc
                pts = np.array([
                    [mouth_cx - 13, mouth_cy - 1],
                    [mouth_cx - 6,  mouth_cy + 5],
                    [mouth_cx,      mouth_cy + 7],
                    [mouth_cx + 6,  mouth_cy + 5],
                    [mouth_cx + 13, mouth_cy - 1],
                ], np.int32)
                cv2.polylines(frame, [pts], False, color, 2, aa)

            elif emotion == "laughing":
                cv2.ellipse(frame, (mouth_cx, mouth_cy + 2), (16, 12),
                            0, 0, 360, color, 2, aa)
                cv2.ellipse(frame, (mouth_cx, mouth_cy + 2), (14, 10),
                            0, 0, 360, dark, -1)

            elif emotion in ("sad",):
                # Frown arc
                pts = np.array([
                    [mouth_cx - 13, mouth_cy + 3],
                    [mouth_cx - 6,  mouth_cy - 2],
                    [mouth_cx,      mouth_cy - 4],
                    [mouth_cx + 6,  mouth_cy - 2],
                    [mouth_cx + 13, mouth_cy + 3],
                ], np.int32)
                cv2.polylines(frame, [pts], False, color, 2, aa)

            elif emotion == "surprised":
                cv2.circle(frame, (mouth_cx, mouth_cy + 2), 8, color, 2, aa)
                cv2.circle(frame, (mouth_cx, mouth_cy + 2), 6, dark, -1)

            elif emotion == "angry":
                pts = np.array([
                    [mouth_cx - 13, mouth_cy + 2],
                    [mouth_cx - 5,  mouth_cy - 1],
                    [mouth_cx + 5,  mouth_cy - 1],
                    [mouth_cx + 13, mouth_cy + 2],
                ], np.int32)
                cv2.polylines(frame, [pts], False, color, 2, aa)

            elif emotion == "sarcastic":
                # One-sided smirk
                pts = np.array([
                    [mouth_cx - 10, mouth_cy + 1],
                    [mouth_cx,      mouth_cy],
                    [mouth_cx + 10, mouth_cy - 4],
                ], np.int32)
                cv2.polylines(frame, [pts], False, color, 2, aa)

            else:
                # Neutral: flat line
                cv2.line(frame, (mouth_cx - 12, mouth_cy),
                                (mouth_cx + 12, mouth_cy), color, 2, aa)

    def _draw_label(self, frame, hc):
        name  = self.name.upper()
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        tw, _ = cv2.getTextSize(name, font, scale, 1)[0]
        tx    = self.cx - tw // 2
        ty    = self.ground + 28
        cv2.putText(frame, name, (tx, ty), font, scale, self.color, 1, cv2.LINE_AA)

    def _draw_speaking_dots(self, frame, hc, t):
        """Three bouncing dots above head to indicate active speech."""
        dot_y = hc[1] - HEAD_RADIUS - 20
        for i in range(3):
            dx  = hc[0] - 10 + i * 10
            bob = int(5 * math.sin(t * 9 + i * 1.8))
            cv2.circle(frame, (dx, dot_y + bob), 4, self.color, -1, cv2.LINE_AA)
