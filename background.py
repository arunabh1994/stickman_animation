import cv2
import numpy as np
import math
from config import VIDEO_WIDTH as W, VIDEO_HEIGHT as H, GROUND_Y


def draw_scene(frame: np.ndarray, scene_id: str, t: float):
    """
    Draw the full background for a given scene.
    Currently all scenes use the 'living_room' set.
    Extend with if/elif to add different rooms.
    """
    _draw_living_room(frame, t)


# ─── Living Room ──────────────────────────────────────────────────────────────

def _draw_living_room(frame: np.ndarray, t: float):
    # 1. Wall
    frame[:GROUND_Y, :] = (215, 205, 190)   # Warm cream (BGR)

    # 2. Floor
    frame[GROUND_Y:, :] = (155, 145, 130)

    # 3. Baseboard
    cv2.rectangle(frame, (0, GROUND_Y), (W, GROUND_Y + 14), (130, 118, 105), -1)

    # 4. Window (back wall center)
    _draw_window(frame, t)

    # 5. Sofa (behind characters, in background)
    _draw_sofa(frame)

    # 6. Curtains (animated)
    _draw_curtain(frame, t, "left")
    _draw_curtain(frame, t, "right")

    # 7. Corner plants
    _draw_plant(frame, 55,      GROUND_Y, t, phase=0.0)
    _draw_plant(frame, W - 55,  GROUND_Y, t, phase=math.pi)

    # 8. Floor rug
    _draw_rug(frame)


def _draw_window(frame, t):
    wx, wy = W // 2 - 110, 40
    ww, wh = 220, 260

    # Sky gradient inside window
    for row in range(wh):
        frac  = row / wh
        blue  = int(240 - frac * 30)
        green = int(220 - frac * 20)
        frame[wy + row, wx:wx + ww] = (blue, green, 200)

    # Clouds (drift slowly leftward, wrap)
    for cloud_idx, (base_x, base_y, size) in enumerate([(60, 70, 1.0), (160, 50, 0.7)]):
        drift = (t * 8 + cloud_idx * 80) % (ww + 60) - 30
        cx    = wx + int(base_x - drift) % ww
        cy    = wy + base_y
        _draw_cloud(frame, cx, cy, size)

    # Window frame
    cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), (160, 140, 110), 4)
    cv2.line(frame, (wx + ww // 2, wy), (wx + ww // 2, wy + wh), (160, 140, 110), 3)
    cv2.line(frame, (wx, wy + wh // 2), (wx + ww, wy + wh // 2), (160, 140, 110), 3)

    # Window sill
    cv2.rectangle(frame, (wx - 6, wy + wh), (wx + ww + 6, wy + wh + 12), (170, 150, 120), -1)


def _draw_cloud(frame, cx, cy, scale):
    """Draw a simple 3-puff cloud."""
    r = int(22 * scale)
    for dx, dy, fr in [(0, 0, 1.0), (-r, 6, 0.8), (r, 8, 0.8)]:
        cv2.circle(frame, (cx + dx, cy + dy), int(r * fr), (255, 255, 255), -1)


def _draw_sofa(frame):
    """A simple flat sofa shape."""
    sx, sy = 180, GROUND_Y - 65
    ew     = W - 180
    # Seat
    cv2.rectangle(frame, (sx, sy), (ew, GROUND_Y), (95, 72, 145), -1)
    # Back rest
    cv2.rectangle(frame, (sx, sy - 48), (ew, sy + 8), (82, 62, 130), -1)
    # Arm rests
    cv2.rectangle(frame, (sx, sy - 48), (sx + 42, GROUND_Y), (82, 62, 130), -1)
    cv2.rectangle(frame, (ew - 42, sy - 48), (ew, GROUND_Y), (82, 62, 130), -1)
    # Cushion dividers (lines)
    mid = (sx + ew) // 2
    cv2.line(frame, (mid, sy - 45), (mid, GROUND_Y - 2), (70, 52, 115), 2)


def _draw_curtain(frame, t, side):
    """
    Animated drape curtain.
    Uses a sine-wave polygon edge to simulate fabric folds.
    The entire curtain sways gently back and forth.
    """
    COLOR       = (65, 50, 120)    # Dark maroon/burgundy (BGR)
    FOLD_COLOR  = (50, 38, 90)     # Darker fold shadow
    WIDTH       = 165              # How wide the curtain hangs in from the edge

    sway = 10 * math.sin(t * 0.38 + (0 if side == "left" else math.pi))

    pts = []
    if side == "left":
        for y in range(0, H + 20, 12):
            wave = 22 * math.sin(y * 0.025 + t * 0.25)
            pts.append([max(0, int(WIDTH + sway + wave)), y])
        pts = [[0, 0]] + pts + [[0, H]]
    else:
        for y in range(0, H + 20, 12):
            wave = 22 * math.sin(y * 0.025 + t * 0.25 + math.pi)
            pts.append([min(W, int(W - WIDTH - sway - wave)), y])
        pts = [[W, 0]] + pts + [[W, H]]

    pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts_arr], COLOR)

    # Fold highlight lines for fabric texture
    num_folds = 6
    for fi in range(num_folds):
        fold_pts = []
        y_start  = int(H * fi / num_folds)
        y_end    = int(H * (fi + 1) / num_folds)
        for y in range(y_start, y_end, 8):
            if side == "left":
                wave = 22 * math.sin(y * 0.025 + t * 0.25)
                fx   = max(0, int(WIDTH + sway + wave - 28))
            else:
                wave = 22 * math.sin(y * 0.025 + t * 0.25 + math.pi)
                fx   = min(W, int(W - WIDTH - sway - wave + 28))
            fold_pts.append([fx, y])

        if len(fold_pts) > 1:
            farr = np.array(fold_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [farr], False, FOLD_COLOR, 3, cv2.LINE_AA)

    # Curtain rod
    if side == "left":
        cv2.rectangle(frame, (0, 0), (WIDTH + 40, 16), (130, 100, 80), -1)
        cv2.circle(frame, (WIDTH + 40, 8), 10, (110, 85, 65), -1)
    else:
        cv2.rectangle(frame, (W - WIDTH - 40, 0), (W, 16), (130, 100, 80), -1)
        cv2.circle(frame, (W - WIDTH - 40, 8), 10, (110, 85, 65), -1)


def _draw_plant(frame, cx, floor_y, t, phase=0.0):
    """An animated potted plant with swaying leaves."""
    pot_h, pot_w = 38, 28
    pot_y = floor_y - pot_h

    # Pot
    cv2.rectangle(frame, (cx - pot_w, pot_y), (cx + pot_w, floor_y), (75, 95, 140), -1)
    cv2.line(frame, (cx - pot_w, pot_y), (cx + pot_w, pot_y), (55, 75, 110), 2)

    # Main stem
    cv2.line(frame, (cx, pot_y), (cx, pot_y - 65), (35, 115, 35), 3, cv2.LINE_AA)

    # Leaves (positions relative to stem, each with individual sway phase)
    leaves = [
        (-28, -28, 22, phase + 0.0),
        ( 28, -38, 22, phase + math.pi),
        (-18, -55, 18, phase + 1.2),
        ( 20, -64, 18, phase - 1.2),
        (  0, -75, 20, phase + 0.6),
    ]
    for dx, dy, size, ph in leaves:
        sway      = int(5 * math.sin(t * 0.9 + ph))
        leaf_cx   = cx + dx + sway
        leaf_cy   = pot_y + dy
        cv2.ellipse(frame, (leaf_cx, leaf_cy), (size, size // 2),
                    45, 0, 360, (35, 155, 35), -1)
        cv2.ellipse(frame, (leaf_cx, leaf_cy), (size, size // 2),
                    45, 0, 360, (20, 100, 20), 1, cv2.LINE_AA)


def _draw_rug(frame):
    """A decorative rug on the floor."""
    rug_x, rug_y = W // 2 - 280, GROUND_Y + 14
    rug_w, rug_h = 560, 38
    cv2.rectangle(frame, (rug_x, rug_y), (rug_x + rug_w, rug_y + rug_h), (110, 75, 130), -1)
    # Rug border pattern
    cv2.rectangle(frame, (rug_x + 6, rug_y + 4), (rug_x + rug_w - 6, rug_y + rug_h - 4),
                  (140, 100, 160), 2)
    # Simple pattern lines
    for i in range(6):
        lx = rug_x + 30 + i * 85
        cv2.line(frame, (lx, rug_y + 8), (lx, rug_y + rug_h - 8), (140, 100, 160), 1)
