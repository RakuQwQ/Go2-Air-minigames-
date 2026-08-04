"""
Go2 Dual Control -- Voice + Hand Gesture
=========================================
"""
import asyncio, json, os, sys, time, logging, random, io, math, threading
from datetime import datetime
import queue
from collections import deque

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

# The dog's STA-L IP is fixed at 192.168.12.1
# The per-device AES-128 key for this robot:
ROBOT_IP   = os.environ.get("UNITREE_ROBOT_IP",   "192.168.12.1")
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY", "2efc5b865f5030b88bf65df39af2b36e")
HOLD_SECS = 1.0
STABLE_FRAMES = 5


# ============================================================
# MICROPHONE MANAGER – pick from dropdown + real-time volume meter
# ============================================================
class MicManager:
    """Manages mic selection, live volume monitoring, and re-initialization."""
    def __init__(self):
        import speech_recognition as sr
        self.sr = sr
        self.mic_pairs = []            # [(device_idx, display_name), ...]
        self.current_device_index = -1
        self.selected_mic = None       # sr.Microphone instance
        self.volume_level = 0.0        # 0.0 – 1.0 for the bar display
        self._sample_rate = 16000
        self._audio_thread_running = [True]
        self._volume_lock = threading.Lock()
        self._refresh_list()

    def _refresh_list(self):
        """Scan all devices and keep only real input microphones (skip outputs, duplicates, mappers)."""
        import pyaudio
        try:
            names = self.sr.Microphone.list_microphone_names()
            pa = pyaudio.PyAudio()
            self.mic_pairs = []
            seen_mics = set()  # Track unique mic names to avoid duplicates
            for idx, name in enumerate(names):
                safe = name.encode('ascii', errors='replace').decode()
                # Check if this is a real input device
                try:
                    info = pa.get_device_info_by_index(idx)
                    is_input = info.get('maxInputChannels', 0) > 0
                except Exception:
                    is_input = False
                if not is_input:
                    continue

                lower = safe.lower()

                # Skip the Microsoft Sound Mapper (virtual device at index 0 on Windows)
                if idx == 0 and 'microsoft' in lower:
                    continue

                # Skip devices that are clearly outputs/speakers/headphones
                skip_keywords = ['output', 'speaker', 'headphone']
                if any(kw in lower for kw in skip_keywords):
                    continue

                # Skip devices with mostly garbled/question-mark names
                # (preserve entries like 'Microphone Array (????? Int' which has useful info)
                q_count = safe.count('?')
                if q_count > 0 and len(safe.strip()) > 0:
                    # If most chars are ?, skip it; otherwise keep it
                    alpha_chars = sum(c.isascii() and c.isalpha() for c in safe)
                    if alpha_chars == 0:
                        continue
                # Skip device entries that are just empty or pure symbols
                if safe.strip() == '':
                    continue

                # Skip hostApi >= 3 (WDM-KS) — these often fail to open or are duplicates
                host_api = info.get('hostApi', -1)
                if host_api >= 3:
                    continue

                # Deduplicate: same device name from multiple host APIs (0=MME, 1=WDM-KS, 2=WASAPI)
                # Keep only the one from hostApi 0 (MME, which works best with speech_recognition)
                if host_api > 0:
                    same_name = [p for p in self.mic_pairs if p[1] == safe]
                    if same_name:
                        continue

                self.mic_pairs.append((idx, safe))
            pa.terminate()
        except Exception as e:
            print(f"  Mic list error: {e}")
            self.mic_pairs = []

    def get_mic_names(self):
        """Return list of display strings for the dropdown."""
        return [f"[{idx}] {name}" for idx, name in self.mic_pairs]

    def select_mic(self, list_index):
        """Select a mic by its index in the displayed list. Returns True on success."""
        if 0 <= list_index < len(self.mic_pairs):
            device_idx, name = self.mic_pairs[list_index]
            self.current_device_index = device_idx
            try:
                self.selected_mic = self.sr.Microphone(device_index=device_idx,
                                                       sample_rate=self._sample_rate)
                print(f"  Mic selected: [{device_idx}] {name} @ {self._sample_rate} Hz")
                # Start volume monitor after selection
                self._start_volume_monitor()
                return True
            except Exception as e:
                print(f"  Mic select error: {e}")
                self.selected_mic = None
                return False
        return False

    def _start_volume_monitor(self):
        """Start a daemon thread that continuously monitors mic volume via PyAudio.
        Auto-calibrates the scale factor on first run so talking fills ~50-80% of the bar.
        """
        import pyaudio
        def _monitor():
            if self.current_device_index < 0:
                return
            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=self._sample_rate,
                                input=True,
                                input_device_index=self.current_device_index,
                                frames_per_buffer=1024)
                stream.start_stream()

                # ── Auto-calibrate scale factor ──
                # Collect a baseline of silence RMS for 0.5s, then set scale = baseline * 8
                # This way normal speech (~4x baseline) shows as ~50% on the bar
                baseline_rms = 0
                sample_count = 0
                calibrate_start = time.time()
                while time.time() - calibrate_start < 0.5:
                    try:
                        data = stream.read(512, exception_on_overflow=False)
                        import struct
                        samples = struct.unpack_from('<%dh' % (len(data)//2), data)
                        if samples:
                            rms = math.sqrt(sum(s*s for s in samples)/len(samples))
                            baseline_rms += rms
                            sample_count += 1
                    except Exception:
                        pass
                if sample_count > 0:
                    baseline_rms /= sample_count
                # Scale: baseline * 15 ≈ less sensitive (speech needs to be ~10x louder than silence)
                scale_factor = max(baseline_rms * 15, 100.0)  # minimum scale of 100
                print(f"  Volume meter calibrated: baseline RMS={baseline_rms:.1f},"
                      f" scale={scale_factor:.0f}")

                while self._audio_thread_running[0]:
                    try:
                        data = stream.read(512, exception_on_overflow=False)
                        import struct
                        samples = struct.unpack_from('<%dh' % (len(data)//2), data)
                        if samples:
                            rms = math.sqrt(sum(s*s for s in samples)/len(samples))
                            # Use auto-calibrated scale
                            vol = min(rms / scale_factor, 1.0)
                            with self._volume_lock:
                                self.volume_level = vol
                    except Exception:
                        pass
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception as e:
                print(f"  Volume monitor error: {e}")
        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

    def stop_volume_monitor(self):
        self._audio_thread_running[0] = False

    def get_volume(self):
        """Get current smoothed volume level (0.0 – 1.0)."""
        with self._volume_lock:
            # Apply a little smoothing for the display
            return self.volume_level

    def cleanup(self):
        self.stop_volume_monitor()


# ============================================================
# HAND GESTURE DETECTOR
# ============================================================
class HandGestureDetector:
    def __init__(self):
        import urllib.request
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            print("Downloading hand model...")
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")

        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.HandLandmarker.create_from_options(opts)
        self._stable = None
        self._stable_cnt = 0
        self._hold_start = 0.0
        self._fired = False
        # ── Thumb dial state ──
        self.thumb_dial_active = False    # True when we're in continuous thumb-dial mode
        self.thumb_dial_angle = 0.0       # current selected angle (degrees), snapped to nearest 45°
        self.thumb_dial_direction = ""    # "up" or "down"
        self.thumb_dial_step = 0          # counter to fire commands periodically

    def _extended(self, lm):
        """
        More robust finger-extension check using angle between
        finger base→PIP→TIP vectors.  Returns a dict of booleans.

        For each finger we compare the angle at the PIP joint:
        - If the finger is straight the TIP→PIP→MCP angle is close to 180°
        - If curled the angle is significantly smaller.

        Thumb uses a different check: compare thumb TIP→IP vector
        against a reference to see if it's sticking out to the side.
        """
        import math

        def angle(a, b, c):
            """Angle (degrees) at point b formed by vectors ba and bc."""
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 * n2 == 0:
                return 0.0
            cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
            return math.degrees(math.acos(cos_a))

        def finger_extended(mcp, pip, tip, threshold=140.0):
            """A finger is 'extended' if the PIP angle >= threshold."""
            return angle(mcp, pip, tip) >= threshold

        # Extract (x, y) tuples – use normalised coords (ratio OK for angles)
        def pt(i):
            return (lm[i].x, lm[i].y)

        return {
            "thumb":  finger_extended(pt(1), pt(3), pt(4), threshold=120.0),
            "index":  finger_extended(pt(5), pt(6), pt(8)),
            "middle": finger_extended(pt(9), pt(10), pt(12)),
            "ring":   finger_extended(pt(13), pt(14), pt(16)),
            "pinky":  finger_extended(pt(17), pt(18), pt(20)),
        }

    def _classify(self, lm, handedness_label):
        """
        Classify hand gesture using angle-based finger states +
        handedness (Left / Right) to disambiguate thumb direction.
        """
        e = self._extended(lm)
        cnt = sum(e.values())

        is_left = handedness_label == "Left"

        # ── Open palm (five) ───────────────────────────────
        if cnt >= 4:
            return "five"

        # ── Love / ILY sign (🤟) ─────────────────────────
        # thumb + index + pinky extended, middle + ring curled
        if (e["thumb"] and e["index"] and e["pinky"]
                and not e["middle"] and not e["ring"]):
            return "love"

        # ── Thumb up / thumb down ────────────────────────
        if e["thumb"] and cnt == 1:
            # Thumb tip Y vs middle MCP Y (hand centre):
            #   thumb_tip.y < middle_mcp.y  → thumb is pointing UP
            #   thumb_tip.y > middle_mcp.y  → thumb is pointing DOWN
            # (Y axis is inverted in image coords: 0 = top of frame)
            if lm[4].y < lm[9].y:
                return "thumb_up"
            else:
                return "thumb_down"

        # ── Point (index extended, others curled) ─────────
        # Only index needs to be extended; thumb can be either.
        # Direction is determined by index tip vs hand centre x-position.
        # Frame is mirrored (flipped horizontally), so:
        #   index tip to the RIGHT of centre → "point_fwd" (walk forward)
        #   index tip to the LEFT  of centre → "point_bwd" (walk backward)
        if e["index"] and not e["middle"] and not e["ring"] and not e["pinky"]:
            # Use middle finger MCP (landmark 9) as hand centre reference
            hand_centre_x = lm[9].x
            if lm[8].x > hand_centre_x:
                return "point_fwd"   # pointing right
            else:
                return "point_bwd"  # pointing left

        return None

    def detect(self, frame):
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        current = None
        landmarks = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            landmarks = lm
            # Get handedness label ("Left" or "Right") from the result
            handedness_label = "Right"
            if result.handedness and result.handedness[0]:
                handedness_label = result.handedness[0][0].category_name

            current = self._classify(lm, handedness_label)
            conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
                     (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
            h, w = frame.shape[:2]
            for i, pt in enumerate(lm):
                cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 5, (0,255,0), -1)
            for i, j in conns:
                cv2.line(frame,(int(lm[i].x*w),int(lm[i].y*h)),(int(lm[j].x*w),int(lm[j].y*h)),(0,255,0),2)
            if current:
                cv2.putText(frame, current, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

        # ────────────────────────────────────────────────────
        # THUMB DIAL MODE — position-based angle selection
        # ────────────────────────────────────────────────────
        # If we're in dial mode and the hand is still making a thumb gesture, keep adjusting
        if self.thumb_dial_active:
            if current in ("thumb_up", "thumb_down") and landmarks is not None:
                # Still holding thumb gesture — compute angle from thumb position.
                # The frame has been flipped horizontally for display (mirror view),
                # so we negate dx to match the visual orientation. Without this, the
                # indicator would be mirrored relative to what the user sees.
                thumb_tip = landmarks[4]
                middle_mcp = landmarks[9]
                # Vector from hand centre (middle_mcp) to thumb tip, in normalised coords
                # Negate dx to account for the horizontal flip applied to the display frame
                dx = -(thumb_tip.x - middle_mcp.x)
                dy = thumb_tip.y - middle_mcp.y
                # Compute direction angle of thumb relative to hand centre
                # atan2 returns angle in radians: 0 = right, π/2 = down, -π/2 = up, π = left
                # We rotate so 0 = up (12 o'clock), positive = clockwise
                raw_angle = math.degrees(math.atan2(-dx, -dy))  # thumb direction from centre
                # Normalise to 0-360
                raw_angle = raw_angle % 360
                # Snap to nearest 45°
                self.thumb_dial_angle = round(raw_angle / 45.0) * 45.0
                if self.thumb_dial_angle >= 360:
                    self.thumb_dial_angle = 0.0

                # Fire commands periodically while holding
                self.thumb_dial_step += 1
                if self.thumb_dial_step >= 10:
                    self.thumb_dial_step = 0
                    self._fired = True
                    return f"thumb_dial:{self.thumb_dial_angle:.0f}:{self.thumb_dial_direction}"

                # Draw the radial dial overlaid on the frame
                self._draw_thumb_dial(frame, self.thumb_dial_angle, self.thumb_dial_direction)
                return None
            else:
                # Gesture released — fire the turn
                angle = self.thumb_dial_angle
                direction = self.thumb_dial_direction
                self._reset_thumb_dial()
                self._fired = True
                return f"thumb_dial_fire:{angle:.0f}:{direction}"

        # ── Normal (non-dial) gesture detection ──────────
        if current is None:
            self._stable = None; self._stable_cnt = 0; self._hold_start = 0; self._fired = False
            return None

        # ── Enter thumb dial mode ──
        if current in ("thumb_up", "thumb_down"):
            if landmarks is not None:
                # Once stable and held briefly, enter dial mode
                if current == self._stable:
                    self._stable_cnt += 1
                else:
                    self._stable = current; self._stable_cnt = 1; self._hold_start = 0; self._fired = False
                    return None

                if self._stable_cnt < STABLE_FRAMES:
                    self._draw_progress(frame, 0, "thumb_dial"); return None

                if self._hold_start == 0:
                    self._hold_start = now

                # Enter dial mode after brief hold
                if now - self._hold_start >= 0.6:
                    if not self.thumb_dial_active:
                        # Initialise dial mode
                        self.thumb_dial_active = True
                        self.thumb_dial_direction = current  # "up" or "down"
                        self.thumb_dial_angle = 0.0
                        self.thumb_dial_step = 0
                        print(f"  Thumb dial engaged! Point thumb direction on wheel.")
                    return None
                else:
                    self._draw_progress(frame, (now - self._hold_start) / 0.6, "thumb_dial")
                    return None

        # ── Other gestures (normal hold-to-fire) ──────────
        if current == self._stable:
            self._stable_cnt += 1
        else:
            self._stable = current; self._stable_cnt = 1; self._hold_start = 0; self._fired = False
            return None

        if self._stable_cnt < STABLE_FRAMES:
            self._draw_progress(frame, 0, current); return None

        if self._hold_start == 0: self._hold_start = now

        if now - self._hold_start >= HOLD_SECS:
            if not self._fired: self._fired = True; return current
            return None
        else:
            self._draw_progress(frame, (now - self._hold_start)/HOLD_SECS, current)
            return None

    def _reset_thumb_dial(self):
        self.thumb_dial_active = False
        self.thumb_dial_angle = 0.0
        self.thumb_dial_direction = ""
        self.thumb_dial_step = 0
        self._stable = None
        self._stable_cnt = 0
        self._hold_start = 0
        self._fired = False

    def _draw_thumb_dial(self, frame, angle_deg, direction):
        """Draw a radial dial showing the selected turn angle (0-360°), snapped to 45° increments.
        Angle 0° = forward (no turn), 90° = right, 180° = backward, 270° = left."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 4

        # ── Background circle ──
        cv2.circle(frame, (cx, cy), radius, (40, 40, 50), 2)
        cv2.circle(frame, (cx, cy), radius + 15, (30, 30, 40), 1)

        # ── Tick marks every 45° ──
        for deg in range(0, 360, 45):
            rad = math.radians(deg - 90)
            inner_r = radius - 14
            x1 = int(cx + inner_r * math.cos(rad))
            y1 = int(cy + inner_r * math.sin(rad))
            x2 = int(cx + radius * math.cos(rad))
            y2 = int(cy + radius * math.sin(rad))
            col = (180, 180, 200) if deg == angle_deg else (100, 100, 120)
            thick = 3 if deg == angle_deg else 2
            cv2.line(frame, (x1, y1), (x2, y2), col, thick)

            # Angle labels
            label_r = radius - 28
            lx = int(cx + label_r * math.cos(rad))
            ly = int(cy + label_r * math.sin(rad))
            label = f"{deg}°"
            col_lbl = (0, 220, 255) if deg == angle_deg else (180, 180, 200)
            cv2.putText(frame, label, (lx - 15, ly + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_lbl, 1)

        # ── Selected wedge (filled) ──
        display_angle = angle_deg
        # Convert to signed: 0-180 = right turn, 180-360 = left turn (negative)
        if display_angle > 180:
            signed_angle = display_angle - 360
        else:
            signed_angle = display_angle
        # Draw wedge from 0° (top/forward) to selected angle
        if abs(signed_angle) > 2:
            if signed_angle >= 0:
                pts = [(cx, cy)]
                for a in range(0, int(signed_angle) + 1, 2):
                    rad = math.radians(a - 90)
                    pts.append((int(cx + radius * math.cos(rad)), int(cy + radius * math.sin(rad))))
                cv2.fillPoly(frame, [np.array(pts, np.int32)], (0, 100, 200))
            else:
                pts = [(cx, cy)]
                for a in range(0, int(signed_angle) - 1, -2):
                    rad = math.radians(a - 90)
                    pts.append((int(cx + radius * math.cos(rad)), int(cy + radius * math.sin(rad))))
                cv2.fillPoly(frame, [np.array(pts, np.int32)], (0, 80, 180))

        # ── Direction arrow ──
        arrow_rad = math.radians(signed_angle - 90)
        ax = int(cx + (radius - 18) * math.cos(arrow_rad))
        ay = int(cy + (radius - 18) * math.sin(arrow_rad))
        cv2.arrowedLine(frame, (cx, cy), (ax, ay), (0, 200, 255), 3, tipLength=0.25)

        # ── Center dot ──
        cv2.circle(frame, (cx, cy), 5, (200, 200, 50), -1)
        cv2.putText(frame, "👆", (cx - 7, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

        # ── Labels ──
        if signed_angle > 0:
            turn_txt = f"RIGHT {signed_angle:.0f}°"
        elif signed_angle < 0:
            turn_txt = f"LEFT {abs(signed_angle):.0f}°"
        else:
            turn_txt = "0° (no turn)"
        cv2.putText(frame, f"Turn {turn_txt}",
                    (cx - 80, cy + radius + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "Release thumb to fire!",
                    (cx - 80, cy + radius + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 150), 1)

    def _draw_progress(self, frame, frac, name):
        h,w,_=frame.shape; bw,bh=220,14; x1=(w-bw)//2; y1=h-55
        s = "Stabilizing..." if frac == 0 else f"Hold {name}... {int(frac*100)}%"
        cv2.rectangle(frame,(x1,y1),(x1+bw,y1+bh),(50,50,50),-1)
        cv2.rectangle(frame,(x1,y1),(x1+int(bw*min(frac,1)),y1+bh),(0,220,0),-1)
        cv2.rectangle(frame,(x1,y1),(x1+bw,y1+bh),(180,180,180),1)
        cv2.putText(frame, s, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)


# ============================================================
# MAIN
# ============================================================
async def main():
    print("="*60)
    print("  Go2 Dual Control — Voice + Hand Gesture")
    print("  (laptop mic + webcam)")
    print("="*60)

    print(f"\n  Robot : {ROBOT_IP}")
    print(f"  AES   : {AES_128_KEY[:16]}...\n")

    # ── Microphone Manager ─────────────────────────────────
    global mic_mgr
    mic_mgr = MicManager()
    mic_options = mic_mgr.get_mic_names()

    print(f"  Found {len(mic_options)} microphone(s):")
    for i, opt in enumerate(mic_options):
        print(f"    [{i}] {opt}")

    # Drop-down state (cycling via keyboard)
    selected_mic_idx = [0]  # mutable for closure

    # Auto-select the best available mic (prefer 3.5mm Realtek, fallback to first)
    if mic_options:
        # Try to find a 'Microphone (Realtek' device first (the 3.5mm jack)
        found = False
        for i, opt in enumerate(mic_options):
            if 'realtek' in opt.lower() and 'microphone' in opt.lower() and 'array' not in opt.lower():
                mic_mgr.select_mic(i)
                selected_mic_idx[0] = i
                print(f"  --> Auto-selected: {opt}")
                found = True
                break
        if not found:
            mic_mgr.select_mic(0)
            print(f"  --> Auto-selected: {mic_options[0]}")
    else:
        print("  ⚠ No microphones found! Voice control disabled.")

    print("  Voice commands:")
    print("    forward / go       → Walk forward")
    print("    backward / back    → Walk backward")
    print("    turn / spin        → Turn 180°")
    print("    stop / halt        → Stop all movement")
    print("    attack / get 'em   → Front pounce (hop on someone)")
    print("    good boy / dance   → Dance party!")
    print("    up / stand         → Stand up")
    print("    rest / sit         → Sit")
    print("    laydown / lie      → Lay down")
    print("    jump               → Front jump")
    print("    hello / wave       → Say hello")
    print()
    print("  Gestures (hold ~1s to your webcam):")
    print("    Open palm              → Stop")
    print("    Thumb up/down          → Enter turn dial (move thumb L/R)")
    print("    Point index to RIGHT   → Walk forward")
    print("    Point index to LEFT    → Walk backward")
    print("    Love sign (🤟 ILY)     → Finger Heart")
    print()
    print("  Turn Dial: Hold thumb up/down, then point your thumb in the")
    print("    direction you want the dog to turn. Snaps to nearest 45°.")
    print("    Point forward (⬆)  → 0°   (no turn)")
    print("    Point right (➡)    → 90°  (turn right)")
    print("    Point back (⬇)     → 180° (turn around)")
    print("    Point left (⬅)     → 270° (turn left)")
    print("    Release thumb to fire the turn command.")
    print("  Startup sequence: BalanceStand → StandUp (with response verification)")
    print("\n  Ctrl+C to exit.\n")

    cmd_map = {
        "stand":(SPORT_CMD["StandUp"],None),"sit":(SPORT_CMD["Sit"],None),
        "laydown":(SPORT_CMD["StandDown"],None),"jump":(SPORT_CMD["FrontJump"],None),
        "hello":(SPORT_CMD["Hello"],None),"five":(SPORT_CMD["StopMove"],None),
        "forward":(SPORT_CMD["Move"],{"x":0.3,"y":0,"z":0}),
        "backward":(SPORT_CMD["Move"],{"x":-0.3,"y":0,"z":0}),
        "turn180":(SPORT_CMD["Move"],{"x":0,"y":0,"z":3.14}),  # π radians ≈ 180° turn
        "attack":(SPORT_CMD["FrontPounce"],None),
        "dance":(SPORT_CMD["Dance1"],None),
        "love":(SPORT_CMD["FingerHeart"],None),
        "point_fwd":(SPORT_CMD["Move"],{"x":0.3,"y":0,"z":0}),   # index pointed right → walk forward
        "point_bwd":(SPORT_CMD["Move"],{"x":-0.3,"y":0,"z":0}),  # index pointed left  → walk backward
    }
    names = {
        "stand":"Stand up","sit":"Sit down","laydown":"Lay down","jump":"Jump",
        "hello":"Hello!","five":"Stop","forward":"Walk forward",
        "backward":"Walk backward","turn180":"Turn 180°",
        "attack":"Attack!","dance":"Dance!","love":"Finger Heart",
        "point_fwd":"Point RIGHT → Walk forward",
        "point_bwd":"Point LEFT → Walk backward",
    }

    # ── Connect ────────────────────────────────────────────
    print("  Connecting to Go2 ...")
    try:
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA,
                                       ip=ROBOT_IP,
                                       aes_128_key=AES_128_KEY)
        await conn.connect()
    except Exception as e:
        print(f"\n  Connection error: {e}")
        return

    dc = None
    for i in range(30):
        await asyncio.sleep(0.5)
        if conn.datachannel and conn.datachannel.channel and conn.datachannel.channel.readyState == "open":
            dc = conn.datachannel.channel
            break
    if not dc:
        print("\n  Could not open data channel.")
        await conn.disconnect()
        return

    print("  Connected! Verifying sport mode activation...")

    # ── Auto-stop timer for velocity commands ──────────────
    # Move with velocity params (x/y/z) makes the dog keep moving
    # until StopMove is sent. We auto-stop after MOVEMENT_DURATION seconds
    # to prevent the dog from being "soft-locked" (stuck in a movement state).
    MOVEMENT_DURATION = 1.5           # how long a velocity Move runs
    _movement_stop_task = None        # asyncio.Task for the auto-stop delay
    MAX_STARTUP_RETRIES = 3           # max attempts for each startup command

    def _is_movement_cmd(api_id):
        """Returns True if this is a velocity-based Move command that needs auto-stop."""
        return api_id == SPORT_CMD["Move"]

    async def _auto_stop_after(delay):
        """After `delay` seconds, send StopMove to prevent soft-lock."""
        await asyncio.sleep(delay)
        await send_cmd(SPORT_CMD["StopMove"], wait_for_response=False)
        print(f"  [Auto-stop] Movement timed out after {delay}s")

    def _cancel_auto_stop():
        """Cancel any pending auto-stop timer."""
        nonlocal _movement_stop_task
        if _movement_stop_task is not None and not _movement_stop_task.done():
            _movement_stop_task.cancel()
            _movement_stop_task = None

    def _schedule_auto_stop():
        """Cancel any previous auto-stop and schedule a new one."""
        nonlocal _movement_stop_task
        _cancel_auto_stop()
        _movement_stop_task = asyncio.ensure_future(_auto_stop_after(MOVEMENT_DURATION))

    def gen_id():
        return int(datetime.now().timestamp()*1000%2147483648)+random.randint(0,999)

    # ── Shared reference to the pub_sub for sport command responses ──
    pub_sub = conn.datachannel.pub_sub

    async def send_cmd(api_id, param=None, wait_for_response=True, timeout=3.0):
        """Send a sport command and optionally wait for a response.

        Uses the data channel's pub_sub system which matches responses
        to requests by a generated UUID. Returns the parsed response
        dict if wait_for_response=True, otherwise None.
        """
        req_id = gen_id()
        payload = {
            "header": {
                "identity": {
                    "id": req_id,
                    "api_id": api_id
                }
            },
            "parameter": json.dumps(param or api_id)
        }
        try:
            if wait_for_response:
                response = await asyncio.wait_for(
                    pub_sub.publish("rt/api/sport/request", payload, "msg"),
                    timeout=timeout
                )
                if _is_movement_cmd(api_id):
                    _schedule_auto_stop()
                else:
                    _cancel_auto_stop()
                return response
            else:
                pub_sub.publish_without_callback("rt/api/sport/request", payload, "msg")
                if _is_movement_cmd(api_id):
                    _schedule_auto_stop()
                else:
                    _cancel_auto_stop()
                return None
        except asyncio.TimeoutError:
            print(f"  ⚠ Command [api_id={api_id}] timed out after {timeout}s (no response)")
            return None
        except (ConnectionError, BrokenPipeError, AttributeError) as e:
            print(f"  ⚠ Dog disconnected: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            print(f"  Send ERROR ({type(e).__name__}): {e}")
            return None

    async def _send_startup_cmd(api_id, label):
        """Send a startup command and retry on failure. Returns True on success."""
        for attempt in range(1, MAX_STARTUP_RETRIES + 1):
            print(f"  Sending {label} (attempt {attempt}/{MAX_STARTUP_RETRIES})...")
            resp = await send_cmd(api_id, wait_for_response=True, timeout=5.0)
            if resp is not None:
                # Got a response — the dog acknowledged the command
                resp_topic = resp.get("topic", "")
                resp_id = resp.get("data", {}).get("header", {}).get("identity", {})
                print(f"  ✅ {label} confirmed (topic={resp_topic})")
                return True
            else:
                print(f"  ⚠ {label} failed or timed out on attempt {attempt}")
                if attempt < MAX_STARTUP_RETRIES:
                    wait = attempt * 1.5
                    print(f"    Retrying in {wait:.0f}s...")
                    await asyncio.sleep(wait)
        print(f"  ❌ {label} failed after {MAX_STARTUP_RETRIES} attempts")
        return False

    # ── Startup: BalanceStand → StandUp ──
    # BalanceStand (cmd 1002) activates the sport mode controller,
    # required before any Move/Dance/etc commands will work.
    success = await _send_startup_cmd(SPORT_CMD["BalanceStand"], "BalanceStand (activate sport mode)")
    if success:
        await asyncio.sleep(1.0)
        success = await _send_startup_cmd(SPORT_CMD["StandUp"], "StandUp")
    else:
        print("  ⚠ Sport mode activation failed — StandUp may not work.")
        await _send_startup_cmd(SPORT_CMD["StandUp"], "StandUp (best-effort)")

    if success:
        print("  ✅ Dog is standing up and sport mode is active!")
    else:
        print("  ⚠ Startup may not have completed successfully. Proceeding anyway...")
    await asyncio.sleep(1.5)

    # ── Webcam ─────────────────────────────────────────────
    gesture = HandGestureDetector()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("  Webcam not found!")
        await conn.disconnect()
        return

    # Use higher resolution for a bigger window
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("  Webcam ready! (640x480)\n")

    # ── Voice thread ───────────────────────────────────────
    vq = queue.Queue()
    vr = [True]
    current_mic_ref = [mic_mgr.selected_mic]  # shared mutable reference
    # Keep the last 5 transcribed words for on-screen display
    recent_words = deque(maxlen=5)

    def voice_thread():
        import speech_recognition as sr
        r = sr.Recognizer()
        # Use adaptive thresholding so the recognizer continuously adjusts
        # to the ambient noise level (works better with varying mic sensitivity).
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        # Wait for a mic to become available
        while vr[0] and current_mic_ref[0] is None:
            time.sleep(0.3)
        if not vr[0]:
            return
        # Keep a persistent source open for the entire session
        source_obj = current_mic_ref[0]
        try:
            source_obj.__enter__()
        except Exception:
            pass
        # Calibrate ambient noise, then apply a small bump (1.3x) to reduce
        # false triggers from background hum — but keep it adaptive.
        try:
            r.adjust_for_ambient_noise(source_obj, duration=1.5)
            r.energy_threshold = max(r.energy_threshold * 3.0, 200)
            print(f"  Voice threshold: {r.energy_threshold:.0f} (adaptive)")
        except Exception as e:
            print(f"  Voice calibration error: {e}")
            r.energy_threshold = 300  # default fallback
        kw = {
            "forward": ["forward", "go forward", "go", "move", "straight"],
            "backward": ["backward", "back", "back up", "reverse"],
            "turn180": ["turn", "turn around", "spin", "rotate", "180"],
            "five":    ["stop", "halt", "freeze", "whoa", "brake"],
            "attack": ["attack", "get em", "get 'em", "get them", "pounce", "sic 'em", "sicem"],
            "dance": ["good boy", "good girl", "dance", "party", "boogie"],
            "stand":   ["up","up now","standing"],
            "sit":     ["rest","rest now","resting","wrestle"],
            "laydown": ["lay down","laydown","laid down","lie down","lying down"],
            "jump":    ["jump","jumping"],
            "hello":   ["hello","hello there","hey low","hollow"],
        }
        while vr[0]:
            try:
                audio = r.listen(source_obj, timeout=1.0, phrase_time_limit=2.0)
                text = r.recognize_google(audio).lower().strip()
                safe_text = text.encode('ascii', errors='replace').decode()
                recent_words.append(safe_text)
                print(f"  [Voice heard] {text}")
                for cmd, trigs in kw.items():
                    for t in trigs:
                        if t in text or text in t:
                            vq.put(cmd)
                            break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"  [Voice API error] {e}")
                continue
            except Exception as e:
                print(f"  [Voice error] {e}")
                continue
        try:
            source_obj.__exit__(None, None, None)
        except Exception:
            pass
    t = threading.Thread(target=voice_thread, daemon=True)
    t.start()

    last_action = ""
    last_action_time = 0

    # ── Clickable dropdown state ──
    dropdown_open = [False]   # whether the dropdown list is shown
    MIC_DROPDOWN_MAX_VISIBLE = 8  # max items visible before scrolling
    frame_size = [640, 480]   # updated each frame

    # Mouse callback for the OpenCV window
    cv_window_name = "Go2 Dual Control"

    def on_mouse(event, x, y, flags, param):
        nonlocal dropdown_open, selected_mic_idx, mic_options, current_mic_ref, frame_size
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = frame_size[0], frame_size[1]
            sel_w, sel_h = 260, 28
            sel_x = w - sel_w - 10
            sel_y = 8
            # Check if click is on the mic selector bar (top-right)
            if sel_x <= x <= sel_x + sel_w and sel_y <= y <= sel_y + sel_h:
                dropdown_open[0] = not dropdown_open[0]
                return

            # If dropdown is open, check if click is on one of the dropdown items
            if dropdown_open[0] and mic_options:
                item_h = 22
                list_x = sel_x
                list_y = sel_y + sel_h + 22  # below volume bar
                list_w = sel_w
                num_items = len(mic_options)
                visible = min(num_items, MIC_DROPDOWN_MAX_VISIBLE)
                total_h = visible * item_h

                if list_x <= x <= list_x + list_w and list_y <= y <= list_y + total_h:
                    clicked_idx = (y - list_y) // item_h
                    if 0 <= clicked_idx < num_items:
                        selected_mic_idx[0] = clicked_idx
                        mic_mgr.select_mic(clicked_idx)
                        current_mic_ref[0] = mic_mgr.selected_mic
                        print(f"  Switched to mic: {mic_options[clicked_idx]}")
                dropdown_open[0] = False
                return

            # Click elsewhere closes dropdown
            dropdown_open[0] = False

    cv2.namedWindow(cv_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cv_window_name, 960, 720)
    cv2.setMouseCallback(cv_window_name, on_mouse)

    # ── Timestamp of the last thumb_dial Move sent (to throttle) ──
    _last_dial_move_time = 0.0

    try:
        while True:
            # ── Check voice queue ──
            try:
                vc = vq.get_nowait()
                aid, p = cmd_map.get(vc, (None, None))
                if aid is not None:
                    n = names.get(vc, vc)
                    print(f"  Voice: {n}")
                    last_action = n
                    last_action_time = time.time()
                    if p is not None:
                        await send_cmd(aid, p)
                    else:
                        await send_cmd(aid)
            except queue.Empty:
                pass

            # ── Process webcam frame ──
            ret, frame = cam.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Run hand detection on every frame
                trig = gesture.detect(frame)

                # ── Handle thumb dial payloads ──
                if trig and isinstance(trig, str) and trig.startswith("thumb_dial"):
                    parts = trig.split(":")
                    if parts[0] == "thumb_dial_fire":
                        # User released the thumb gesture — send turn command
                        angle_deg = float(parts[1])
                        direction = parts[2]
                        # Convert 0-360 to signed angle (-180 to 180)
                        if angle_deg > 180:
                            signed_angle = angle_deg - 360
                        else:
                            signed_angle = angle_deg
                        # Map signed angle to z-rotation (-1.0 to 1.0)
                        z_val = max(-1.0, min(1.0, signed_angle / 180.0))
                        if abs(signed_angle) < 5:
                            # Near 0° = no turn, nothing to send
                            pass
                        else:
                            n = f"Turn {'RIGHT' if signed_angle > 0 else 'LEFT'} {abs(signed_angle):.0f}°"
                            print(f"  Gesture dial fire: {n} (z={z_val:.2f})")
                            last_action = n
                            last_action_time = time.time()
                            await send_cmd(SPORT_CMD["Move"], {"x": 0, "y": 0, "z": z_val})
                            # Don't auto-stop here — gesture will release and we want the
                            # turn to execute. The auto-stop from send_cmd will handle it.
                    elif parts[0] == "thumb_dial":
                        # During dial — show live preview (but throttle to avoid flooding)
                        now = time.time()
                        if now - _last_dial_move_time < 0.15:
                            # Skip this frame to avoid flooding commands
                            pass
                        else:
                            _last_dial_move_time = now
                            angle_deg = float(parts[1])
                            direction = parts[2]
                            if angle_deg > 180:
                                signed_angle = angle_deg - 360
                            else:
                                signed_angle = angle_deg
                            z_val = max(-1.0, min(1.0, signed_angle / 180.0))
                            if abs(signed_angle) < 5:
                                n = "0° (no turn)"
                            else:
                                n = f"Dial: {abs(signed_angle):.0f}° {'RIGHT' if signed_angle > 0 else 'LEFT'}"
                            last_action = n
                            last_action_time = time.time()
                            # Send periodic updates to dog for live preview
                            await send_cmd(SPORT_CMD["Move"], {"x": 0, "y": 0, "z": z_val})

                # ── Handle regular gestures ──
                elif trig:
                    aid, p = cmd_map.get(trig, (None, None))
                    if aid is not None:
                        n = names.get(trig, trig)
                        print(f"  Gesture: {n}")
                        last_action = n
                        last_action_time = time.time()
                        if p is not None:
                            await send_cmd(aid, p)
                        else:
                            await send_cmd(aid)

                h, w = frame.shape[:2]
                frame_size[0], frame_size[1] = w, h

                # ── Mic selector bar (top-right, clickable) ──
                sel_w, sel_h = 260, 28
                sel_x = w - sel_w - 10
                sel_y = 8
                # Background (highlight if dropdown open)
                bg_col = (60, 60, 80) if dropdown_open[0] else (40, 40, 40)
                cv2.rectangle(frame, (sel_x, sel_y), (sel_x + sel_w, sel_y + sel_h), bg_col, -1)
                cv2.rectangle(frame, (sel_x, sel_y), (sel_x + sel_w, sel_y + sel_h), (100, 100, 100), 1)
                # Current mic name (truncated if too long)
                if mic_options:
                    label = mic_options[selected_mic_idx[0]]
                    if len(label) > 28:
                        label = label[:26] + ".."
                    cv2.putText(frame, f"  {label}", (sel_x + 6, sel_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
                # Dropdown arrow indicator
                arrow = "v" if not dropdown_open[0] else "^"
                cv2.putText(frame, arrow, (sel_x + sel_w - 16, sel_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

                # ── Volume meter (below selector) ──
                bar_x = sel_x
                bar_y = sel_y + sel_h + 6
                bar_w = sel_w
                bar_h = 14
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
                vol = mic_mgr.get_volume() if mic_options else 0.0
                fill_w = int(bar_w * vol)
                if vol < 0.5:
                    col = (0, int(200 * vol * 2), 50)
                elif vol < 0.75:
                    col = (0, 200, 255)
                else:
                    col = (0, 50, 255)
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x + 1, bar_y + 1),
                                  (bar_x + fill_w, bar_y + bar_h - 1), col, -1)
                for tick_pct in [25, 50, 75]:
                    tx = bar_x + int(bar_w * tick_pct / 100)
                    cv2.line(frame, (tx, bar_y), (tx, bar_y + bar_h), (60, 60, 60), 1)

                # ── Dropdown list (if open) ──
                if dropdown_open[0] and mic_options:
                    item_h = 22
                    list_x = sel_x
                    list_y = sel_y + sel_h + 22
                    list_w = sel_w
                    num_items = len(mic_options)
                    max_vis = MIC_DROPDOWN_MAX_VISIBLE
                    visible = min(num_items, max_vis)
                    total_h = visible * item_h

                    # Background for the list
                    cv2.rectangle(frame, (list_x, list_y),
                                  (list_x + list_w, list_y + total_h), (45, 45, 55), -1)
                    cv2.rectangle(frame, (list_x, list_y),
                                  (list_x + list_w, list_y + total_h), (120, 120, 120), 1)

                    for i in range(visible):
                        item_y = list_y + i * item_h
                        # Highlight the currently selected mic
                        if i == selected_mic_idx[0]:
                            cv2.rectangle(frame, (list_x + 1, item_y),
                                          (list_x + list_w - 1, item_y + item_h),
                                          (70, 90, 130), -1)
                        # Hover effect on mouseover would need tracking, skip for simplicity
                        txt = mic_options[i]
                        if len(txt) > 30:
                            txt = txt[:28] + ".."
                        cv2.putText(frame, txt, (list_x + 6, item_y + 17),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 220), 1)

                    # Scroll hint if there are more items
                    if num_items > max_vis:
                        cv2.putText(frame, f"... ({num_items - max_vis} more)",
                                    (list_x + 6, list_y + total_h - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

                # Overlay last action on the video feed
                if last_action and time.time() - last_action_time < 3:
                    cv2.putText(frame, f">> {last_action}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                # Show the last 5 transcribed words (bottom-left)
                for i, word in enumerate(recent_words):
                    y = frame.shape[0] - 20 - i * 22
                    cv2.putText(frame, f"  {word}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.putText(frame, "Click mic=select  q=quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 220, 150), 1)
                cv2.imshow(cv_window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m') or key == ord('M'):
                    # Cycle microphone selection
                    if mic_options:
                        selected_mic_idx[0] = (selected_mic_idx[0] + 1) % len(mic_options)
                        mic_mgr.select_mic(selected_mic_idx[0])
                        current_mic_ref[0] = mic_mgr.selected_mic
                        print(f"  Switched to mic: {mic_options[selected_mic_idx[0]]}")
                        dropdown_open[0] = False

            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  Shutting down ...")

    # ── Safety: stop the dog before disconnect ──
    _cancel_auto_stop()
    await send_cmd(SPORT_CMD["StopMove"], wait_for_response=False)
    await asyncio.sleep(0.3)

    vr[0] = False
    if mic_mgr:
        mic_mgr.cleanup()
    cam.release()
    cv2.destroyAllWindows()
    await conn.disconnect()
    print("  Done!")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
