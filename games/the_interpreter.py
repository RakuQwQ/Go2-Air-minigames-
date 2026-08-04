"""
interpreter_dynamic.py — Motion Gesture Control for Go2 Air
============================================================
Detects hand gestures via webcam and sends commands to your Go2 Air robot.
DYNAMIC VERSION — self-calibrates centroid thresholds per user.

Gesture Mapping:
  ┌── TWO-HAND STATIC POSES ──────────────────────────────┐
  │ Open hand     + Open hand          → Lay Down         │
  │ Open hand     + Thumbs up          → Stand Up         │
  │ Open hand     + Thumbs down        → Sit              │
  │ Two hands pointing up              → Stop             │
  ├── TWO-HAND MOTION (fist anchor + other hand moves) ──┤
  │ Fist (still) + hand thrust TOWARD  → Walk Forward    │
  │ Fist (still) + hand pull AWAY      → Walk Backward   │
  │ Fist (still) + other hand LEFT     → Turn Left 45°   │
  │ Fist (still) + other hand RIGHT    → Turn Right 45°  │
  └───────────────────────────────────────────────────────┘

Two-hand poses: hold both hands still for a moment.
Motion commands: keep one fist still, move the other hand.

Requires:
  - Go2 Air powered on, PC on WiFi 2 (STA-L mode, 192.168.12.x)
  - AES-128 key
  - Webcam connected to your PC
"""

import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime
import random
from collections import deque
import io

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ── Encoding fix for Chinese Windows ──
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Patch aiortc — skip video/audio transceivers (Go2 SDP has none) ──
from aiortc import RTCPeerConnection
_orig_add = RTCPeerConnection.addTransceiver
def _patched_add(self, kind, **kwargs):
    if kind in ('audio', 'video'):
        return None
    return _orig_add(self, kind, **kwargs)
RTCPeerConnection.addTransceiver = _patched_add

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
    SPORT_CMD,
)

# === CONFIG ===
ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY", "2efc5b865f5030b88bf65df39af2b36e")
GESTURE_COOLDOWN = 1.5
STABLE_FRAMES = 4
MOTION_HISTORY_LEN = 8  # How many frames to track for motion direction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
# HAND CLASSIFICATION — Dynamic 2D Distance-Based Detection
# ═══════════════════════════════════════════════════════════
#
# Uses TWO independent metrics for each finger and classifies
# using distances in 2D (angle, ratio) space:
#
# 1. PIP joint angles (scale-invariant, person-independent)
#    - Straight finger: ~150-180°
#    - Curled finger:   ~60-110°
#
# 2. Fingertip-to-palm distance ratio
#    (fingertip distance from wrist) / (palm size: wrist to middle MCP)
#    - Extended finger: ratio > 2.0
#    - Curled finger:   ratio < 1.4
#
# Each finger is classified by comparing its distance to two ideal
# centroids in normalized (angle, ratio) space:
#   - curled centroid = (0, 0)  — low angle, low ratio
#   - extended centroid = (1, 1) — high angle, high ratio
# The "extendedness" ratio tells us how finger-like vs curled-like it is.
#
# MediaPipe finger landmark indices:
#   Thumb:  1(MCP)  2(PIP)  3(DIP)  4(TIP)
#   Index:  5(MCP)  6(PIP)  7(DIP)  8(TIP)
#   Middle: 9(MCP) 10(PIP) 11(DIP) 12(TIP)
#   Ring:  13(MCP) 14(PIP) 15(DIP) 16(TIP)
#   Pinky: 17(MCP) 18(PIP) 19(DIP) 20(TIP)
# ═══════════════════════════════════════════════════════════

import math
import statistics

# ── Finger states ──
FINGER_CURLED   = 0
FINGER_PARTIAL  = 1
FINGER_EXTENDED = 2

DEBUG_CLASSIFY = True  # Shows per-finger metrics on screen

# ── Dynamic Calibration: per-user centroid learning ──
# These centroids represent the "ideal curled" and "ideal extended" finger
# in (angle_norm, ratio_norm) space. They start at generic defaults and
# are adjusted over time as the system observes confirmed gestures.
# Format: { finger_index: (angle_mean, ratio_mean) }
# Finger indices: 0=index, 1=middle, 2=ring, 3=pinky
_CURLED_CENTROIDS = {
    0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0),
}
_EXTENDED_CENTROIDS = {
    0: (1.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 1.0),
}
# Running data buffers for each finger state
_CURLED_BUFFERS = {i: deque(maxlen=60) for i in range(4)}   # last 60 curled samples
_EXTENDED_BUFFERS = {i: deque(maxlen=60) for i in range(4)} # last 60 extended samples
_CALIBRATION_SAMPLES = 0  # total samples collected


def _update_centroid(finger_idx, angle_norm, ratio_norm, state):
    """
    Feed a classified finger measurement back into the calibration system.
    Updates the running buffer for that finger+state, then recalculates
    the centroid if enough data has been collected.
    """
    global _CALIBRATION_SAMPLES
    buffer = _CURLED_BUFFERS if state == FINGER_CURLED else _EXTENDED_BUFFERS
    centroids = _CURLED_CENTROIDS if state == FINGER_CURLED else _EXTENDED_CENTROIDS
    
    buf = buffer[finger_idx]
    buf.append((angle_norm, ratio_norm))
    _CALIBRATION_SAMPLES += 1
    
    # Recalculate centroid every 10 new samples per finger
    if len(buf) >= 10 and len(buf) % 5 == 0:
        angles = [p[0] for p in buf]
        ratios = [p[1] for p in buf]
        # Use median for robustness against outliers
        centroids[finger_idx] = (statistics.median(angles), statistics.median(ratios))


def _reset_calibration():
    """Reset all learned centroids back to defaults."""
    global _CALIBRATION_SAMPLES
    for i in range(4):
        _CURLED_CENTROIDS[i] = (0.0, 0.0)
        _EXTENDED_CENTROIDS[i] = (1.0, 1.0)
        _CURLED_BUFFERS[i].clear()
        _EXTENDED_BUFFERS[i].clear()
    _CALIBRATION_SAMPLES = 0


def _angle_between(p1, p2, p3):
    """Interior angle at p2 (degrees) formed by vectors p1→p2 and p2→p3."""
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = (v1[0]**2 + v1[1]**2) ** 0.5
    mag2 = (v2[0]**2 + v2[1]**2) ** 0.5
    if mag1 < 0.001 or mag2 < 0.001:
        return 180.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def _classify_finger(angle, tip_ratio, finger_idx=0):
    """
    Classify a SINGLE finger's state using 2D distance-based decision
    with DYNAMIC centroids that adapt per user.
    
    Returns: FINGER_CURLED (0), FINGER_PARTIAL (1), or FINGER_EXTENDED (2).
    """
    # Normalize angle into [0, 1]: 60°→0.0, 180°→1.0
    angle_norm = max(0.0, min(1.0, (angle - 60.0) / 120.0))
    # Normalize ratio into [0, 1]: 0.8→0.0, 3.0→1.0
    ratio_norm = max(0.0, min(1.0, (tip_ratio - 0.8) / 2.2))
    
    # Get the current centroids for this specific finger
    curled_c = _CURLED_CENTROIDS.get(finger_idx, (0.0, 0.0))
    extended_c = _EXTENDED_CENTROIDS.get(finger_idx, (1.0, 1.0))
    
    # Euclidean distance to curled vs extended centroids
    dist_to_curled = math.sqrt(
        (angle_norm - curled_c[0])**2 + (ratio_norm - curled_c[1])**2
    )
    dist_to_extended = math.sqrt(
        (angle_norm - extended_c[0])**2 + (ratio_norm - extended_c[1])**2
    )
    
    total = dist_to_curled + dist_to_extended
    if total < 0.001:
        return FINGER_PARTIAL
    
    # extendedness: 0.0 = curled-like, 1.0 = extended-like
    extendedness = dist_to_curled / total
    
    # Determine state
    if extendedness > 0.65:
        state = FINGER_EXTENDED
    elif extendedness < 0.30:
        state = FINGER_CURLED
    else:
        state = FINGER_PARTIAL
    
    # Feed back into calibration (only confident classifications)
    if state != FINGER_PARTIAL:
        _update_centroid(finger_idx, angle_norm, ratio_norm, state)
    
    return state


def classify_hand(lm):
    """
    Classify hand pose using per-finger classification with DYNAMIC centroids,
    then match the overall finger profile against gesture templates.
    
    Returns: 'open', 'fist', 'thumbs_up', 'thumbs_down', 'point_up',
             'point_forward', 'point_away', or None.
    """
    wrist = lm[0]
    palm_size = ((lm[9].x - wrist.x) ** 2 + (lm[9].y - wrist.y) ** 2) ** 0.5
    if palm_size < 0.001:
        palm_size = 0.001

    def tip_dist(tip_idx):
        d = ((lm[tip_idx].x - wrist.x) ** 2 + (lm[tip_idx].y - wrist.y) ** 2) ** 0.5
        return d / palm_size

    # -- Compute angles and distance ratios for each finger --
    index_angle  = _angle_between(lm[5],  lm[6],  lm[7])
    idx_ratio    = tip_dist(8)
    
    middle_angle = _angle_between(lm[9],  lm[10], lm[11])
    mid_ratio    = tip_dist(12)
    
    ring_angle   = _angle_between(lm[13], lm[14], lm[15])
    ring_ratio   = tip_dist(16)
    
    pinky_angle  = _angle_between(lm[17], lm[18], lm[19])
    pnk_ratio    = tip_dist(20)
    
    thumb_angle  = _angle_between(lm[0],  lm[1],  lm[2])
    thb_ratio    = tip_dist(4)

    # ── Classify each finger with dynamic centroids ──
    # finger_idx: 0=index, 1=middle, 2=ring, 3=pinky
    idx_state  = _classify_finger(index_angle,  idx_ratio,  finger_idx=0)
    mid_state  = _classify_finger(middle_angle, mid_ratio,  finger_idx=1)
    ring_state = _classify_finger(ring_angle,   ring_ratio, finger_idx=2)
    pnk_state  = _classify_finger(pinky_angle,  pnk_ratio,  finger_idx=3)
    
    # ── Thumb analysis (different geometry, not dynamically calibrated) ──
    thumb_extended = thb_ratio > 1.6 and thumb_angle > 100
    thumb_raised = lm[4].y < lm[2].y

    # ── Count finger states ──
    four_curled   = sum(1 for s in [idx_state, mid_state, ring_state, pnk_state] if s == FINGER_CURLED)
    four_extended = sum(1 for s in [idx_state, mid_state, ring_state, pnk_state] if s == FINGER_EXTENDED)
    four_partial  = sum(1 for s in [idx_state, mid_state, ring_state, pnk_state] if s == FINGER_PARTIAL)

    # ── Store debug info ──
    if DEBUG_CLASSIFY:
        state_names = {0: "C", 1: "~", 2: "E"}
        # Show current centroids for reference
        c0 = _CURLED_CENTROIDS.get(0, (0,0))
        e0 = _EXTENDED_CENTROIDS.get(0, (1,1))
        lm[0]._debug_data = dict(
            index_a=int(index_angle),  mid_a=int(middle_angle),
            ring_a=int(ring_angle),    pnk_a=int(pinky_angle),
            thumb_a=int(thumb_angle),
            index_r=round(idx_ratio, 1),  mid_r=round(mid_ratio, 1),
            ring_r=round(ring_ratio, 1),  pnk_r=round(pnk_ratio, 1),
            thumb_r=round(thb_ratio, 1),
            states=f"{state_names[idx_state]}{state_names[mid_state]}{state_names[ring_state]}{state_names[pnk_state]}",
            curled=str(four_curled), extended=str(four_extended),
            # Calibration info
            cal_samples=str(_CALIBRATION_SAMPLES),
            cc=round(c0[0], 2), ce=round(e0[0], 2),  # curled/extended angle centroid
        )

    # ═══════════════════════════════════════════════════════════
    # GESTURE CLASSIFICATION — Decision Tree
    # ═══════════════════════════════════════════════════════════

    # ── THUMBS UP / DOWN (highest priority — must beat fist) ──
    # Thumb extended with curled fingers. Allow up to 1 partial finger
    # since a loose thumbs-up may have slightly bent pinky/ring.
    if thumb_extended and four_extended == 0 and four_curled >= 2:
        if thumb_raised:
            return "thumbs_up"
        else:
            return "thumbs_down"

    # ── FIST: all 4 non-thumb fingers curled, zero extended ──
    # Only if thumb is NOT extended (prevents fist vs thumbs confusion)
    if not thumb_extended and four_extended == 0 and four_curled >= 3:
        return "fist"
    if not thumb_extended and four_extended == 0 and four_partial <= 1 and four_curled >= 2:
        return "fist"

    # ── OPEN HAND: at least 3 non-thumb fingers extended ──
    if four_extended >= 3:
        return "open"
    if four_extended >= 2 and four_curled == 0:
        return "open"

    # ── POINTING ──
    if (idx_state == FINGER_EXTENDED and 
        mid_state == FINGER_CURLED and 
        ring_state == FINGER_CURLED and 
        pnk_state == FINGER_CURLED):
        dx = lm[8].x - lm[5].x
        dy = lm[8].y - lm[5].y
        if abs(dy) > abs(dx) and dy < 0:
            return "point_up"
        elif abs(dy) > abs(dx) and dy > 0:
            return "point_down"
        elif abs(dx) > abs(dy) and dx > 0:
            return "point_forward"
        else:
            return "point_away"

    return None


# ═══════════════════════════════════════════════════════════
# POSE DETECTOR
# ═══════════════════════════════════════════════════════════

class PoseDetector:
    """
    Detects two-hand static poses and two-hand motion gestures.
    
    Two-hand static poses (hold both hands still):
      - Open hand + Open hand          → Lay Down
      - Open hand + Thumbs up          → Stand Up
      - Open hand + Thumbs down        → Sit
      - Both hands pointing up         → Stop
    
    Two-hand motion gestures (fist anchor + other hand moves):
      - Fist (still) + hand TOWARD     → Walk Forward
      - Fist (still) + hand AWAY       → Walk Backward
      - Fist (still) + hand LEFT       → Turn Left 45°
      - Fist (still) + hand RIGHT      → Turn Right 45°
    """

    def __init__(self):
        import urllib.request
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            print("  Downloading hand model (~15 MB)...")
            urllib.request.urlretrieve(url, model_path)
            print("  Model downloaded!")

        base = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.HandLandmarker.create_from_options(opts)

        # Cooldown
        self.last_trigger_time = 0.0

        # ── Fist-anchor motion tracking ──
        # When we detect a fist + another hand, we track the other hand's
        # position over time to determine motion direction.
        self.fist_anchor_active = False   # are we tracking fist-anchor motion?
        self.fist_anchor_other_is_left = None  # True if moving hand is left, False if right
        self.fist_anchor_track = deque(maxlen=MOTION_HISTORY_LEN)  # (x, y) positions of moving hand
        self.fist_anchor_size_track = deque(maxlen=MOTION_HISTORY_LEN)  # hand size of moving hand

        # Two-hand pose counters (must be after fist-anchor init)
        self._reset_pose_counters()

    def _reset_pose_counters(self):
        self.both_open_stable = 0
        self.open_thumbs_up_stable = 0
        self.open_thumbs_down_stable = 0
        self.both_point_up_stable = 0
        self.both_open_triggered = False
        self.open_thumbs_up_triggered = False
        self.open_thumbs_down_triggered = False
        self.both_point_up_triggered = False
        # Also reset fist-anchor motion tracking
        self.fist_anchor_active = False
        self.fist_anchor_other_is_left = None
        self.fist_anchor_track.clear()
        self.fist_anchor_size_track.clear()

    # ── helpers ──────────────────────────────────────────

    def _get_left_right(self, result, labels):
        left = right = None
        for i, lm in enumerate(result.hand_landmarks):
            if i < len(labels):
                if labels[i] == "Left":
                    left = lm
                elif labels[i] == "Right":
                    right = lm
        return left, right

    def _hand_size(self, lm):
        """Estimate apparent hand size using wrist-to-middle-fingertip distance."""
        return ((lm[0].x - lm[12].x) ** 2 + (lm[0].y - lm[12].y) ** 2) ** 0.5

    def _dist(self, a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def _draw_hands(self, frame, result, labels, w, h):
        conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                 (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for i, lm in enumerate(result.hand_landmarks):
            for pti, pt in enumerate(lm):
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 5, (0, 255, 0), -1)
            for ci, cj in conns:
                cv2.line(frame,
                         (int(lm[ci].x * w), int(lm[ci].y * h)),
                         (int(lm[cj].x * w), int(lm[cj].y * h)),
                         (0, 255, 0), 2)
            if i < len(labels):
                wrist = lm[0]
                cv2.putText(frame, labels[i],
                            (int(wrist.x * w) - 10, int(wrist.y * h) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # ── pose matchers ────────────────────────────────────

    def _has_open(self, lc, rc):
        """At least one hand is 'open'."""
        return lc == "open" or rc == "open"

    def _has_thumbs_up(self, lc, rc):
        """At least one hand is 'thumbs_up'."""
        return lc == "thumbs_up" or rc == "thumbs_up"

    def _has_thumbs_down(self, lc, rc):
        """At least one hand is 'thumbs_down'."""
        return lc == "thumbs_down" or rc == "thumbs_down"

    def _both_open(self, lc, rc):
        """Both hands are 'open'."""
        return lc == "open" and rc == "open"

    def _both_point_up(self, lc, rc):
        """Both hands are 'point_up'."""
        return lc == "point_up" and rc == "point_up"

    # ── main detect ──────────────────────────────────────

    def detect(self, frame):
        """Returns (action_key, display_text) or (None, None)."""
        now = time.time()
        if now - self.last_trigger_time < GESTURE_COOLDOWN:
            return None, None

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if not result.hand_landmarks:
            self._reset_pose_counters()
            return None, None

        num_hands = len(result.hand_landmarks)
        labels = []
        if result.handedness:
            for hl in result.handedness:
                if hl:
                    labels.append(hl[0].category_name)

        self._draw_hands(frame, result, labels, w, h)

        # ── Show debug angles + ratios + finger states on screen ──
        debug_y = 100
        for i, lm in enumerate(result.hand_landmarks):
            dd = getattr(lm[0], '_debug_data', None)
            if dd:
                label = labels[i] if i < len(labels) else f"Hand{i}"
                # Line 1: angles
                txt_a = f"{label}: A:{dd['index_a']}°/{dd['mid_a']}°/{dd['ring_a']}°/{dd['pnk_a']}° T:{dd['thumb_a']}°"
                cv2.putText(frame, txt_a, (10, debug_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
                debug_y += 13
                # Line 2: ratios
                txt_r = f"     R:{dd['index_r']}/{dd['mid_r']}/{dd['ring_r']}/{dd['pnk_r']} T:{dd['thumb_r']}"
                cv2.putText(frame, txt_r, (10, debug_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 100), 1)
                debug_y += 13
                # Line 3: finger states (C=curled, ~=partial, E=extended) + calibration
                cal_info = ""
                if 'cal_samples' in dd:
                    cal_info = f"  cal#{dd['cal_samples']} Cc:{dd['cc']} Ec:{dd['ce']}"
                txt_s = f"     {dd['states']}  C:{dd['curled']} E:{dd['extended']}{cal_info}"
                cv2.putText(frame, txt_s, (10, debug_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 50), 1)
                debug_y += 18

        # ═══════════════════════════════════════════════════
        # TWO-HAND DETECTION
        # ═══════════════════════════════════════════════════

        if num_hands >= 2 and len(labels) >= 2:
            left_lm, right_lm = self._get_left_right(result, labels)

            if left_lm is not None and right_lm is not None:
                lc = classify_hand(left_lm)
                rc = classify_hand(right_lm)
                d = self._dist(left_lm[0], right_lm[0])

                cv2.putText(frame, f"L: {lc}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                cv2.putText(frame, f"R: {rc}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                cv2.putText(frame, f"Dist: {d:.2f}", (w - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

                # ──────────────────────────────────────────
                # STATIC TWO-HAND POSES (checked BEFORE fist-anchor)
                # ──────────────────────────────────────────
                # These must come first. Especially thumbs up/down + open hand,
                # because the thumbs hand has curled fingers and could be mistaken
                # for a fist by the motion detector below.

                # Open + Open → Lay Down
                if self._both_open(lc, rc):
                    self.both_open_stable += 1
                    if self.both_open_stable >= STABLE_FRAMES and not self.both_open_triggered:
                        self._reset_pose_counters()
                        self.both_open_triggered = True
                        self.last_trigger_time = now
                        return "laydown", "Open + Open → Lay Down!"
                    cv2.putText(frame, "LAY DOWN POSE", (w//2 - 100, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return None, None
                else:
                    self.both_open_stable = 0
                    self.both_open_triggered = False

                # Open + Thumbs up → Stand Up
                # Also: if one hand is thumbs_up and the other isn't doing anything
                # problematic (fist, open, or nothing), treat as stand
                has_thumbs_up = lc == "thumbs_up" or rc == "thumbs_up"
                other_is_open = (lc == "open" or rc == "open")
                both_thumbs_up = lc == "thumbs_up" and rc == "thumbs_up"
                if has_thumbs_up and (other_is_open or both_thumbs_up):
                    self.open_thumbs_up_stable += 1
                    if self.open_thumbs_up_stable >= STABLE_FRAMES and not self.open_thumbs_up_triggered:
                        self._reset_pose_counters()
                        self.open_thumbs_up_triggered = True
                        self.last_trigger_time = now
                        return "stand", "Open + Thumbs Up → Stand Up!"
                    cv2.putText(frame, "STAND UP POSE", (w//2 - 100, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return None, None
                else:
                    self.open_thumbs_up_stable = 0
                    self.open_thumbs_up_triggered = False

                # Open + Thumbs down → Sit
                has_thumbs_down = lc == "thumbs_down" or rc == "thumbs_down"
                if has_thumbs_down and other_is_open:
                    self.open_thumbs_down_stable += 1
                    if self.open_thumbs_down_stable >= STABLE_FRAMES and not self.open_thumbs_down_triggered:
                        self._reset_pose_counters()
                        self.open_thumbs_down_triggered = True
                        self.last_trigger_time = now
                        return "sit", "Open + Thumbs Down → Sit!"
                    cv2.putText(frame, "SIT POSE", (w//2 - 80, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return None, None
                else:
                    self.open_thumbs_down_stable = 0
                    self.open_thumbs_down_triggered = False

                # Two hands pointing up → Stop
                if self._both_point_up(lc, rc):
                    self.both_point_up_stable += 1
                    if self.both_point_up_stable >= STABLE_FRAMES and not self.both_point_up_triggered:
                        self._reset_pose_counters()
                        self.both_point_up_triggered = True
                        self.last_trigger_time = now
                        return "stop", "Two Point Up → Stop!"
                    cv2.putText(frame, "STOP POSE", (w//2 - 80, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return None, None
                else:
                    self.both_point_up_stable = 0
                    self.both_point_up_triggered = False

                # ──────────────────────────────────────────
                # FIST-ANCHOR MOTION DETECTION
                # One hand is a fist (anchor), the other moves
                # ──────────────────────────────────────────
                # IMPORTANT: thumbs up/down are NOT fists, so they won't trigger this
                is_fist_anchor = (lc == "fist") != (rc == "fist")  # exactly one fist
                if is_fist_anchor:
                    # Identify which hand is the fist (anchor) and which is moving
                    if lc == "fist":
                        anchor_lm = left_lm
                        moving_lm = right_lm
                        moving_is_left = False
                    else:
                        anchor_lm = right_lm
                        moving_lm = left_lm
                        moving_is_left = True

                    moving_cls = classify_hand(moving_lm)

                    # The moving hand can be open, pointing, thumbs-up/down
                    # Track its position and apparent size over time
                    if not self.fist_anchor_active:
                        # Start tracking
                        self.fist_anchor_active = True
                        self.fist_anchor_other_is_left = moving_is_left
                        self.fist_anchor_track.clear()
                        self.fist_anchor_size_track.clear()
                        self.fist_anchor_track.append((moving_lm[0].x, moving_lm[0].y))
                        self.fist_anchor_size_track.append(self._hand_size(moving_lm))
                        cv2.putText(frame, "Fist anchor - move other hand!",
                                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        return None, None
                    else:
                        # Continuing tracking
                        self.fist_anchor_track.append((moving_lm[0].x, moving_lm[0].y))
                        self.fist_anchor_size_track.append(self._hand_size(moving_lm))

                        if len(self.fist_anchor_track) >= 5:
                            # Analyze motion
                            positions = list(self.fist_anchor_track)
                            sizes = list(self.fist_anchor_size_track)

                            # --- Depth change (size) → Forward / Backward ---
                            avg_size = sum(sizes) / len(sizes)
                            if avg_size > 0.001:
                                drift = sizes[-1] - sizes[0]
                                rel_drift = drift / avg_size
                                if rel_drift > 0.25:  # Hand got bigger → toward cam
                                    self._reset_pose_counters()
                                    self.last_trigger_time = now
                                    return "forward", "Fist anchor + thrust TOWARD → Walk Forward!"
                                elif rel_drift < -0.25:  # Hand got smaller → away from cam
                                    self._reset_pose_counters()
                                    self.last_trigger_time = now
                                    return "backward", "Fist anchor + pull AWAY → Walk Backward!"

                            # --- Horizontal movement (x position) → Turn Left / Right ---
                            xs = [p[0] for p in positions]
                            x_drift = xs[-1] - xs[0]
                            if abs(x_drift) > 0.08:  # Significant horizontal movement
                                if x_drift > 0:  # Moved right
                                    self._reset_pose_counters()
                                    self.last_trigger_time = now
                                    return "turn_right", "Fist anchor + hand RIGHT → Turn Right!"
                                else:  # Moved left
                                    self._reset_pose_counters()
                                    self.last_trigger_time = now
                                    return "turn_left", "Fist anchor + hand LEFT → Turn Left!"

                            cv2.putText(frame, "Fist anchor - track motion...",
                                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                            return None, None
                        else:
                            cv2.putText(frame, f"Fist anchor - tracking {len(self.fist_anchor_track)}/5",
                                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                            return None, None
                else:
                    # Not a fist-anchor pair → reset tracking
                    self.fist_anchor_active = False
                    self.fist_anchor_track.clear()
                    self.fist_anchor_size_track.clear()
            else:
                # Could not determine left/right → reset
                self._reset_pose_counters()
        else:
            # Less than 2 hands → reset everything
            self._reset_pose_counters()

        return None, None


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  interpreter_dynamic — Motion Gesture Control for Go2 Air [DYNAMIC]")
    print("=" * 60)

    if not AES_128_KEY:
        print("\n  ERROR: AES key not set!")
        print("  Run: unitree-fetch-aes-key --email your@email.com --region cn --device-type Go2")
        print('  Then: $env:UNITREE_AES_128_KEY = "<32-hex-chars>"')
        sys.exit(1)

    print(f"\n  Robot IP : {ROBOT_IP}")
    print(f"  Cooldown  : {GESTURE_COOLDOWN}s\n")

    print("  GESTURES (show to your webcam):")
    print("  ┌── TWO-HAND STATIC POSES ───────────────────────────────┐")
    print("  │ Open hand     + Open hand          → Lay Down          │")
    print("  │ Open hand     + Thumbs up          → Stand Up          │")
    print("  │ Open hand     + Thumbs down        → Sit               │")
    print("  │ Two hands pointing up              → Stop              │")
    print("  ├── TWO-HAND MOTION (fist anchor + other hand moves) ────┤")
    print("  │ Fist (still)  + hand thrust TOWARD → Walk Forward      │")
    print("  │ Fist (still)  + hand pull AWAY     → Walk Backward     │")
    print("  │ Fist (still)  + other hand LEFT    → Turn Left 45°     │")
    print("  │ Fist (still)  + other hand RIGHT   → Turn Right 45°    │")
    print("  └────────────────────────────────────────────────────────┘")
    print()
    print("  Tips: hold two-hand poses still for a moment.")
    print("  For motion: keep one fist still, move the other hand.")
    print("  Press 'q' to quit. Ctrl+C to exit.\n")

    # ── Connect ──
    print("  >> Connecting to Go2 Air...")
    try:
        conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY,
        )
        await conn.connect()
    except Exception as e:
        print(f"\n  >> Connection failed: {type(e).__name__}: {e}")
        print("  Check: power, WiFi, AES key, phone app disconnected.")
        return

    print("  >> Waiting for data channel...")
    dc = None
    for i in range(30):
        await asyncio.sleep(0.5)
        if conn.datachannel and conn.datachannel.channel and conn.datachannel.channel.readyState == "open":
            dc = conn.datachannel.channel
            break
    if not dc:
        print("\n  >> Could not open data channel. Disconnect phone app.")
        await conn.disconnect()
        return
    print("  >> Connected! ✅\n")

    def gen_id():
        return int(datetime.now().timestamp() * 1000 % 2147483648) + random.randint(0, 999)

    def send_cmd(api_id, param=None):
        cmd = json.dumps({
            "type": "msg",
            "topic": "rt/api/sport/request",
            "data": {
                "header": {"identity": {"id": gen_id(), "api_id": api_id}},
                "parameter": json.dumps(param or api_id),
            },
        })
        try:
            dc.send(cmd)
            return True
        except Exception:
            return False

    # ── Start standing ──
    print("  >> Standing up...")
    send_cmd(SPORT_CMD["StandUp"])
    await asyncio.sleep(2)

    # ── Action map ──
    action_map = {
        "forward":   (SPORT_CMD["Move"],      {"x": 0.3, "y": 0.0, "z": 0.0}, "Walk Forward"),
        "backward":  (SPORT_CMD["Move"],      {"x": -0.3, "y": 0.0, "z": 0.0}, "Walk Backward"),
        "turn_left": (SPORT_CMD["Move"],      {"x": 0.0, "y": 0.0, "z": 0.5},  "Turn Left 45°"),
        "turn_right":(SPORT_CMD["Move"],      {"x": 0.0, "y": 0.0, "z": -0.5}, "Turn Right 45°"),
        "laydown":   (SPORT_CMD["StandDown"],  None,                           "Lay Down"),
        "sit":       (SPORT_CMD["Sit"],        None,                           "Sit"),
        "stand":     (SPORT_CMD["StandUp"],    None,                           "Stand Up"),
        "stop":      (SPORT_CMD["StopMove"],   None,                           "Stop"),
    }

    # ── Safe state machine ──
    dog_state = "standing"

    def send_safe(action_key):
        nonlocal dog_state
        info = action_map.get(action_key)
        if not info:
            return
        api_id, params, name = info

        # Motion commands (forward, backward, turn) need the dog standing first
        if action_key in ("forward", "backward", "turn_left", "turn_right"):
            if dog_state in ("sitting", "laying"):
                send_cmd(SPORT_CMD["StandUp"])
                time.sleep(1.5)
            if dog_state == "moving":
                send_cmd(SPORT_CMD["StopMove"])
                time.sleep(0.3)
            send_cmd(api_id, params)
            dog_state = "moving"

        elif action_key == "laydown":
            if dog_state == "moving":
                send_cmd(SPORT_CMD["StopMove"])
                time.sleep(0.3)
            if dog_state == "sitting":
                send_cmd(SPORT_CMD["StandUp"])
                time.sleep(1.5)
            send_cmd(api_id, params)
            dog_state = "laying"

        elif action_key == "sit":
            if dog_state == "moving":
                send_cmd(SPORT_CMD["StopMove"])
                time.sleep(0.3)
            if dog_state == "laying":
                send_cmd(SPORT_CMD["StandUp"])
                time.sleep(1.5)
            send_cmd(api_id, params)
            dog_state = "sitting"

        elif action_key in ("stand"):
            if dog_state == "moving":
                send_cmd(SPORT_CMD["StopMove"])
                time.sleep(0.3)
            if dog_state in ("sitting", "laying"):
                print(f"  >> Dog is {dog_state}, standing up first...")
                send_cmd(SPORT_CMD["StandUp"])
                time.sleep(1.5)
            send_cmd(api_id, params)
            dog_state = "standing"

        elif action_key == "stop":
            send_cmd(api_id, params)
            dog_state = "standing"

        else:
            send_cmd(api_id, params)

    # ── Webcam ──
    detector = PoseDetector()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("  >> Could not open webcam!")
        await conn.disconnect()
        return
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("  >> Webcam ready!\n")

    last_text = ""
    last_time = 0
    move_start = 0
    MOVE_TIMEOUT = 3.0

    try:
        while True:
            now = time.time()
            ret, frame = cam.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            key, display = detector.detect(frame)

            if key and display:
                print(f"  >> {display}")
                if key in ("forward", "backward", "turn_left", "turn_right"):
                    move_start = now
                send_safe(key)
                if key != "stop":
                    if key == "forward":
                        last_text = "Walk Forward"
                    elif key == "backward":
                        last_text = "Walk Backward"
                    elif key == "turn_left":
                        last_text = "Turn Left 45°"
                    elif key == "turn_right":
                        last_text = "Turn Right 45°"
                    else:
                        info = action_map.get(key)
                        if info:
                            last_text = info[2]
                    last_time = now

            # Auto-stop
            if dog_state == "moving" and now - move_start > MOVE_TIMEOUT:
                send_cmd(SPORT_CMD["StopMove"])
                dog_state = "standing"
                last_text = "Auto-Stopped"
                last_time = now

            # Overlay
            if last_text and now - last_time < 3:
                cv2.putText(frame, f">> {last_text}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cd = GESTURE_COOLDOWN - (now - detector.last_trigger_time)
            if cd > 0:
                cv2.putText(frame, f"Cooldown: {cd:.1f}s", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

            cv2.putText(frame, "Gesture Control | q=quit",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 150), 1)
            cv2.imshow("interpreter_dynamic - Go2 Gesture Control [DYNAMIC]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n  >> Shutting down...")

    cam.release()
    cv2.destroyAllWindows()
    await conn.disconnect()
    print("  >> Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
