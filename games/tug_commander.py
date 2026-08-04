"""
Better Commander - Go2 Air Game
================================
A two-player hand gesture accuracy duel.

HOW IT WORKS:
  1. The webcam captures both players simultaneously
  2. Player A = left half of frame, Player B = right half of frame
  3. Each player performs a hand gesture; the system detects which one
     is "more accurate" (higher number of landmarks detected + more extended fingers)
  4. The more accurate player scores 1 point
  5. The scale moves: +1 toward Player A, -1 toward Player B
  6. Robot dog WALKS FORWARD toward whichever player just scored
  7. Robot dog TURNS to face the most recently scored player
  8. Initially, dog faces the Y direction (perpendicular to the two players)
  9. Scale range: -3 (Player B wins)  ← · · · ·  +3 (Player A wins)

GESTURES DETECTED:
  - thumbs_up  : 🖕
  - open_palm  : 🖐
  - pointing   : ☝
  - peace      : ✌
  - fist       : ✊
  - call_me    : 🤙
  - rock_on    : 🤘

Prerequisites:
  - Python 3.8+
  - `pip install mediapipe opencv-python numpy`
  - Go2 Air powered on, computer connected to the dog's WiFi
  - UNITREE_AES_128_KEY environment variable set (per-device AES key)
"""

import asyncio
import json
import os
import sys
import time
import random
import logging
from datetime import datetime
from enum import Enum

# ── Force UTF-8 for stdout to prevent cp950 emoji crashes ──
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# =============================================================================
# IMPORTS
# =============================================================================
# CRITICAL PATCH: Override aiortc's addTransceiver BEFORE importing unitree_webrtc_connect
# The Go2 Air (Chinese firmware) SDP answer has NO media lines (only data channel).
# Without this patch, aiortc crashes because it expects video/audio transceivers
# that the Go2 never provides. This is the #1 fix for "dog did not move at all".
from aiortc import RTCPeerConnection
_orig_add_transceiver = RTCPeerConnection.addTransceiver
def _patched_add_transceiver(self, kind, **kwargs):
    if kind in ('audio', 'video'):
        return None
    return _orig_add_transceiver(self, kind, **kwargs)
RTCPeerConnection.addTransceiver = _patched_add_transceiver

# PATCH: Prevent emoji crash in unitree_webrtc_connect's print_status
# The library uses emoji characters that crash on cp950 (Chinese Windows).
# We patch the print function to strip non-ASCII characters if needed.
import unitree_webrtc_connect.util
_orig_print_status = unitree_webrtc_connect.util.print_status
def _safe_print_status(status_type, status_message):
    """Safe version that strips emoji for cp950 compatibility."""
    try:
        _orig_print_status(status_type, status_message)
    except UnicodeEncodeError:
        # Fallback: ASCII-safe print
        now_str = datetime.now().strftime("%H:%M:%S")
        safe_type = status_type.encode('ascii', errors='replace').decode('ascii')
        safe_msg = status_message.encode('ascii', errors='replace').decode('ascii')
        print(f"[{now_str}] {safe_type:<25}: {safe_msg:<15}")
unitree_webrtc_connect.util.print_status = _safe_print_status

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
    SPORT_CMD,
)

# =============================================================================
# CONFIG
# =============================================================================
ROBOT_IP = "192.168.12.1"
AES_128_KEY = "2efc5b865f5030b88bf65df39af2b36e"

# Game settings
WIN_SCORE = 3           # scale goes from -WIN_SCORE to +WIN_SCORE
GESTURE_COOLDOWN = 1.0  # seconds between round evaluations
ROUND_WAIT_TIME = 5.0   # forced pause after each round before next detection
PREP_TIME = 10.0        # seconds to wait after first gesture is detected so players can adjust
MOVE_DURATION = 1.0     # how long the dog walks forward each round (seconds)
TURN_DURATION = 1.6     # how long the dog rotates to face player (seconds) — 90 deg at TURN_SPEED
WALK_SPEED = 0.50       # forward walk speed (m/s) — 0.5m/s * 1.0s = 0.5m = 50cm
CMD_INTERVAL = 0.1      # how often to re-send velocity commands (Go2 needs continuous stream)
MAX_CMD_RETRIES = 3     # max retries for a single command before considering dog offline
TURN_ANGLE = 1.57       # 90 degrees in radians (pi/2) — turn to face the scoring player
TURN_SPEED = 1.2        # rotation speed (rad/s) — slightly faster for crisp turns
TURN_OVERSCAN = 1.15    # multiplier to ensure full turn (15% extra) — compensates for accel/decel
BACKWARD_SPEED = -0.40  # backward walk speed (m/s) — negative = backward movement

# Camera regions — Player A is left half, Player B is right half
FRAME_WIDTH = 640
FRAME_HEIGHT = 720

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# GESTURE SCORER
# =============================================================================

class GestureScorer:
    """
    Analyzes a hand in a given image region and returns:
      - gesture name (str or None)
      - accuracy score (float) — higher = more clearly defined gesture
    """
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            print("  Downloading hand model (~15 MB)...")
            urllib.request.urlretrieve(url, model_path)
            print("  Model downloaded!")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,  # detect up to 2 hands
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def get_finger_states(lm):
        """Determine which fingers are extended."""
        dx = lm[4].x - lm[5].x
        dy = lm[4].y - lm[5].y
        thumb_up = (dx * dx + dy * dy) ** 0.5 > 0.05
        return {
            "thumb":  thumb_up,
            "index":  lm[8].y  < lm[6].y,
            "middle": lm[12].y < lm[10].y,
            "ring":   lm[16].y < lm[14].y,
            "pinky":  lm[20].y < lm[18].y,
        }

    @staticmethod
    def classify_gesture(f):
        """Classify a gesture from finger states."""
        if f["thumb"] and not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            return "thumbs_up"
        if all(f[k] for k in ["thumb", "index", "middle", "ring", "pinky"]):
            return "open_palm"
        if not f["thumb"] and f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            return "pointing"
        if not f["thumb"] and f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
            return "peace"
        if not any(f[k] for k in ["thumb", "index", "middle", "ring", "pinky"]):
            return "fist"
        if f["thumb"] and f["index"] and not f["middle"] and not f["ring"] and f["pinky"]:
            return "call_me"
        if not f["thumb"] and f["index"] and not f["middle"] and not f["ring"] and f["pinky"]:
            return "rock_on"
        return None

    def analyze_hand(self, landmarks):
        """
        Given MediaPipe landmarks for one hand, return:
          (gesture_name, accuracy_score)
        where accuracy_score is based on landmark clarity and finger extension.
        """
        f = self.get_finger_states(landmarks)
        gesture = self.classify_gesture(f)

        # Compute accuracy: a combination of how many fingers are
        # confidently extended + how spread out the hand landmarks are.
        extended_count = sum(1 for v in f.values() if v)

        # Landmark spread: standard deviation of landmark positions
        # Higher spread = more open/clear hand pose
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        spread = np.std(xs) + np.std(ys)

        # Confidence bonus: gestures with recognizable patterns score higher
        # A clearly classified gesture is better than an ambiguous one
        if gesture is not None:
            # Recognized gesture: base score from extended fingers + spread
            score = 2.0 + extended_count * 0.5 + spread * 10.0
        else:
            # Unrecognized hand shape: partial score based on detection quality
            score = 0.5 + spread * 5.0

        return gesture, score

    def detect_and_score(self, frame, roi_x_start, roi_x_end):
        """
        Analyze a specific region of the frame for hand detection.

        Args:
            frame: full BGR frame
            roi_x_start: left bound (as fraction 0.0–1.0 or pixel)
            roi_x_end: right bound

        Returns:
            (gesture_name, accuracy_score, annotated_landmarks_list)
        """
        h, w, _ = frame.shape

        # Convert ROI bounds to pixels if fractions
        if roi_x_start < 1.0:
            x1 = int(roi_x_start * w)
        else:
            x1 = int(roi_x_start)
        if roi_x_end < 1.0:
            x2 = int(roi_x_end * w)
        else:
            x2 = int(roi_x_end)

        x1 = max(0, x1)
        x2 = min(w, x2)

        # Crop the ROI from the frame first — smaller image = faster detection
        roi_frame = frame[:, x1:x2]

        # Downscale for detection speed (MediaPipe handles 256px nicely)
        detect_h = 256
        scale = detect_h / roi_frame.shape[0]
        detect_w = int(roi_frame.shape[1] * scale)
        detect_frame = cv2.resize(roi_frame, (detect_w, detect_h))

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        # Map landmarks back to original frame coordinates
        inv_scale_x = roi_frame.shape[1] / detect_w
        inv_scale_y = roi_frame.shape[0] / detect_h

        best_gesture = None
        best_score = -1.0
        best_landmarks = None

        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                # Map wrist back to full-frame coords
                wrist_x = int(hand_lm[0].x * detect_w * inv_scale_x + x1)
                if x1 <= wrist_x <= x2:
                    gesture, score = self.analyze_hand(hand_lm)
                    if score > best_score:
                        best_score = score
                        best_gesture = gesture
                        # Transform landmarks back to full-frame coordinates
                        transformed = []
                        for pt in hand_lm:
                            transformed.append(type('LM', (), {
                                'x': (pt.x * detect_w * inv_scale_x + x1) / w,
                                'y': (pt.y * detect_h * inv_scale_y) / h,
                                'z': pt.z
                            })())
                        best_landmarks = transformed

        return best_gesture, best_score, best_landmarks

    def detect_both(self, frame, roi_a, roi_b):
        """
        Detect hands for BOTH players in a single MediaPipe pass on the
        composite frame, returning results for each ROI.
        More efficient than calling detect_and_score twice.
        """
        h, w, _ = frame.shape
        x1_a, x2_a = roi_a
        x1_b, x2_b = roi_b

        # Downscale composite for faster detection
        detect_h = 256
        scale = detect_h / h
        detect_w = int(w * scale)
        detect_frame = cv2.resize(frame, (detect_w, detect_h))

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        inv_scale_x = w / detect_w
        inv_scale_y = h / detect_h

        # Check each detected hand against both ROIs
        p1_gesture, p1_score, p1_lm = None, -1.0, None
        p2_gesture, p2_score, p2_lm = None, -1.0, None

        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                wrist_x_detect = hand_lm[0].x * detect_w
                wrist_x_full = int(wrist_x_detect * inv_scale_x)

                gesture, score = self.analyze_hand(hand_lm)

                # Transform landmarks to full-frame coordinates
                transformed = []
                for pt in hand_lm:
                    transformed.append(type('LM', (), {
                        'x': pt.x * detect_w * inv_scale_x / w,
                        'y': pt.y * detect_h * inv_scale_y / h,
                        'z': pt.z
                    })())

                if x1_a <= wrist_x_full <= x2_a:
                    if score > p1_score:
                        p1_score = score
                        p1_gesture = gesture
                        p1_lm = transformed
                elif x1_b <= wrist_x_full <= x2_b:
                    if score > p2_score:
                        p2_score = score
                        p2_gesture = gesture
                        p2_lm = transformed

        return (p1_gesture, p1_score, p1_lm), (p2_gesture, p2_score, p2_lm)


# =============================================================================
# GAME STATE
# =============================================================================

class Player(Enum):
    A = "A"
    B = "B"


class CameraSetup:
    """
    Stores which physical camera index is used for each player.
    """
    def __init__(self, cam_a: int, cam_b: int):
        self.cam_a = cam_a
        self.cam_b = cam_b

    @property
    def same_camera(self) -> bool:
        return self.cam_a == self.cam_b

    def describe(self) -> str:
        if self.same_camera:
            return f"Both players share Camera {self.cam_a} (split-screen)"
        else:
            return f"Player A = Camera {self.cam_a} | Player B = Camera {self.cam_b}"

    def __repr__(self) -> str:
        return self.describe()


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class SideAssignment:
    """
    Maps Player A and Player B to sides of the camera frame.
    Default: Player A = LEFT half, Player B = RIGHT half.
    The user can choose at startup via an on-screen menu.
    """
    def __init__(self, a_side: Side = Side.LEFT, b_side: Side = Side.RIGHT):
        self.a_side = a_side
        self.b_side = b_side

    def get_side_for_player(self, player: Player) -> Side:
        if player == Player.A:
            return self.a_side
        return self.b_side

    def get_player_for_side(self, side: Side) -> Player:
        if self.a_side == side:
            return Player.A
        return Player.B

    def get_roi(self, player: Player, frame_width: int):
        """Return (x_start, x_end) for the given player's half of the frame."""
        side = self.get_side_for_player(player)
        mid = frame_width // 2
        if side == Side.LEFT:
            return (0, mid)
        else:
            return (mid, frame_width)

    def side_name(self, player: Player) -> str:
        side = self.get_side_for_player(player)
        return side.value.capitalize()

    def describe(self) -> str:
        return f"Player A = {self.side_name(Player.A)} half | Player B = {self.side_name(Player.B)} half"

    def __repr__(self) -> str:
        return self.describe()


class GameState:
    """Tracks the duel between Player A and Player B."""

    def __init__(self):
        self.scale = 0       # -WIN_SCORE to +WIN_SCORE (negative = B winning)
        self.round = 0
        self.last_winner = None   # Player.A or Player.B or None (initial)
        self.last_round_time = 0
        self.game_over = False
        self.winner = None
        self.prep_start_time = 0  # when first gesture was detected (prep phase start)
        self.prep_first_gesture_seen = False  # whether ANY gesture has been seen (starts countdown, does NOT reset)
        self.prep_gestures_detected = False  # whether both gestures have been seen

    def score_point(self, player: Player):
        """Award a point to the given player."""
        if self.game_over:
            return

        self.round += 1
        self.last_winner = player
        # Reset prep state for next round
        self.prep_start_time = 0
        self.prep_first_gesture_seen = False
        self.prep_gestures_detected = False

        if player == Player.A:
            self.scale += 1
        else:
            self.scale -= 1

        # Check win condition
        if self.scale >= WIN_SCORE:
            self.game_over = True
            self.winner = Player.A
        elif self.scale <= -WIN_SCORE:
            self.game_over = True
            self.winner = Player.B

    @property
    def is_waiting(self):
        """Whether we're in the forced wait period between rounds."""
        return (self.last_round_time > 0
                and time.time() - self.last_round_time < ROUND_WAIT_TIME)

    def facing_direction_for_neutral(self):
        """
        Returns the rotation value to face perpendicular to the players.
        Positive = turning toward Player B's side,
        Negative = turning toward Player A's side.
        """
        return 0.0  # Neutral — manually set perpendicular

    def get_rotation_to_player(self, player: Player):
        """
        Return rotation direction to face the given player.
        Positive rotation = turn right (toward Player B).
        Negative rotation = turn left (toward Player A).
        """
        if player == Player.A:
            return -TURN_SPEED  # Turn left to face Player A
        else:
            return TURN_SPEED   # Turn right to face Player B

    def get_move_toward_player(self, player: Player):
        """
        Return movement direction toward the given player.
        Player A is on the left side, Player B on the right side.
        x = forward/backward
        y = strafe left/right
        z = rotation
        """
        if player == Player.A:
            # Walk forward-left (toward Player A's position)
            return {"x": WALK_SPEED, "y": 0.0, "z": 0.0}
        else:
            # Walk forward-right (toward Player B's position)
            return {"x": WALK_SPEED, "y": 0.0, "z": 0.0}

    def get_game_status_text(self):
        """Return descriptive text for the current game state."""
        bar_parts = []
        for i in range(-WIN_SCORE, WIN_SCORE + 1):
            if i == self.scale:
                bar_parts.append("●")
            elif i == 0:
                bar_parts.append("|")
            else:
                bar_parts.append("·")

        bar = "".join(bar_parts)
        a_marks = "█" * max(0, self.scale)
        b_marks = "█" * max(0, -self.scale)

        return (
            f"  Round {self.round}  |  "
            f"Player A [{a_marks}] ← {bar} → [{b_marks}] Player B  |  "
            f"Last: {self.last_winner.name if self.last_winner else '-'}"
        )


# =============================================================================
# ROBOT CONTROLLER
# =============================================================================

class RobotController:
    """Handles all communication with the Go2 Air robot.
    
    FIXES applied (soft-lock prevention):
      1. Continuous velocity streaming: Move commands are re-sent every 100ms
         during turns/walks so the dog actually moves for the full duration.
      2. Command retry: If a send fails, retry up to MAX_CMD_RETRIES times.
      3. Facing angle tracking: Keeps track of dog's cumulative facing angle,
         so restart can reliably return to neutral (perpendicular to players).
      4. Timeout-safe celebrate_win: Won't sleep 4s if dog is offline.
      5. Connected flag toggles off if too many consecutive failures,
         so the game loop never blocks on a dead dog.
    """

    def __init__(self):
        self.conn = None
        self.dc = None
        self.facing_angle = 0.0   # cumulative rotation angle (radians)
        self._consecutive_failures = 0
        self.connected = False
        self._last_turn_direction = None  # +1 (right/B) or -1 (left/A) — last face_player direction
        self._connect_attempts = 0
        self._keepalive_task = None  # async task that pings the dog periodically

    async def connect(self):
        """Establish WebRTC connection to the robot."""
        print("  >> Connecting to Go2 Air...")

        if not AES_128_KEY:
            print("\n  ERROR: AES key not set!")
            print("  Set UNITREE_AES_128_KEY environment variable.")
            return False

        try:
            self.conn = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=ROBOT_IP,
                aes_128_key=AES_128_KEY,
            )
            await self.conn.connect()
            print("  >> UnitreeWebRTCConnection.connect() returned successfully")

            # CRITICAL: Wait for data channel to be fully open before proceeding
            # (same pattern as go2_dual_control.py which works on Chinese firmware)
            self.dc = None
            for i in range(30):
                await asyncio.sleep(0.5)
                if (self.conn.datachannel and self.conn.datachannel.channel
                        and self.conn.datachannel.channel.readyState == "open"):
                    self.dc = self.conn.datachannel.channel
                    print(f"  >> Data channel open (readyState={self.dc.readyState})")
                    break

            if not self.dc:
                print("  Could not open data channel after 15 seconds!")
                return False

            self.connected = True
            self._consecutive_failures = 0
            self.start_keepalive()
            print("  Connected! [OK]")
            return True

        except Exception as e:
            print(f"\n  Connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _gen_id(self):
        return int(datetime.now().timestamp() * 1000 % 2147483648) + random.randint(0, 999)

    def _send_cmd(self, api_id, param=None):
        """Send a command to the robot with retry logic.
        Returns True if the command was sent successfully.
        Tracks consecutive failures and marks self.connected=False if too many."""
        if not self.connected or not self.dc:
            return False
        if self.dc.readyState != "open":
            print(f"  [WARN] Data channel not open (state={self.dc.readyState})")
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_CMD_RETRIES:
                self.connected = False
                print(f"  [X] TOO MANY FAILURES ({MAX_CMD_RETRIES}). Dog marked offline.")
                self.stop_keepalive()
            return False

        cmd = json.dumps({
            "type": "msg",
            "topic": "rt/api/sport/request",
            "data": {
                "header": {"identity": {"id": self._gen_id(), "api_id": api_id}},
                "parameter": json.dumps(param or api_id),
            },
        })
        for attempt in range(MAX_CMD_RETRIES):
            try:
                self.dc.send(cmd)
                self._consecutive_failures = 0  # reset on success
                return True
            except Exception as e:
                if attempt < MAX_CMD_RETRIES - 1:
                    print(f"  [WARN] Send error (attempt {attempt+1}/{MAX_CMD_RETRIES}): {e}")
                    time.sleep(0.05)  # tiny pause before retry
                else:
                    print(f"  [X] Send failed after {MAX_CMD_RETRIES} attempts: {e}")
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CMD_RETRIES:
                        self.connected = False
                        print(f"  [X] TOO MANY FAILURES. Dog marked offline.")
        return False

    def send_raw_move(self, x=0.0, y=0.0, z=0.0):
        """Alias matching go2_dual_control exactly."""
        return self._send_cmd(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

    def stand_up(self):
        """Make the dog stand — use StandUp (1004) same as working examples."""
        return self._send_cmd(SPORT_CMD["StandUp"])

    def sit_down(self):
        """Make the dog sit."""
        return self._send_cmd(SPORT_CMD["Sit"])

    def stop_move(self):
        """Stop all movement."""
        return self._send_cmd(SPORT_CMD["StopMove"])

    def move(self, x=0.0, y=0.0, z=0.0):
        """Move the dog with velocity commands."""
        return self._send_cmd(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

    def celebrate(self):
        """Celebration animation."""
        return self._send_cmd(SPORT_CMD["Dance1"])

    async def _stream_velocity(self, x=0.0, y=0.0, z=0.0, duration=1.0):
        """
        Send a velocity command continuously (every CMD_INTERVAL seconds)
        for the given duration. This is the KEY fix for the Go2's one-shot
        velocity behaviour — the dog needs a continuous stream to keep moving.
        
        Returns True if all sends succeeded, False if any failed.
        """
        end_time = time.time() + duration
        all_ok = True
        while time.time() < end_time:
            ok = self.move(x=x, y=y, z=z)
            if not ok:
                all_ok = False
                # Don't keep sending if dog is offline
                if not self.connected:
                    return False
            await asyncio.sleep(CMD_INTERVAL)
        return all_ok

    async def face_player(self, player: Player):
        """
        Rotate the dog 90 degrees to face the scoring player.
        
        Always a 90-degree turn from the neutral position (perpendicular to players).
        - Player A is left  => negative z rotation (turn left)
        - Player B is right => positive z rotation (turn right)
        
        After this turn, the dog will be facing the player. 
        Call return_to_neutral() afterward to turn back.
        """
        # Stop any previous movement first
        self.stop_move()
        await asyncio.sleep(0.1)

        # Apply overscan to compensate for accel/decel
        effective_angle = TURN_ANGLE * TURN_OVERSCAN

        # Determine direction: negative z = left (Player A), positive z = right (Player B)
        direction = -1.0 if player == Player.A else 1.0
        rotation = direction * TURN_SPEED
        duration = effective_angle / TURN_SPEED

        print(f"  >> Rotating 90 deg {'LEFT' if player == Player.A else 'RIGHT'} (z={rotation:+.2f}) for {duration:.2f}s...")
        await self._stream_velocity(x=0.0, y=0.0, z=rotation, duration=duration)
        self.stop_move()
        # Track direction for return_to_neutral
        self._last_turn_direction = direction
        print(f"  >> Turn done, now facing Player {player.name}")

    async def return_to_neutral(self):
        """
        Rotate the dog 90 degrees back to the neutral position (perpendicular to players).
        The direction is opposite to the last face_player direction.
        Since we don't track absolute orientation, we just turn back 90 degrees
        in the opposite direction. Call this AFTER face_player + walk.
        """
        self.stop_move()
        await asyncio.sleep(0.1)

        # Turn back: 90 degrees in the opposite direction
        # We always turn back the same amount regardless of which direction we came from
        effective_angle = TURN_ANGLE * TURN_OVERSCAN
        duration = effective_angle / TURN_SPEED

        print(f"  >> Returning to neutral (180 deg total from {self._last_turn_direction})...")
        # Reverse the last turn direction
        rotation = -(self._last_turn_direction or 1.0) * TURN_SPEED
        await self._stream_velocity(x=0.0, y=0.0, z=rotation, duration=duration)
        self.stop_move()
        self._last_turn_direction = None
        print(f"  >> Returned to neutral")

    async def walk_toward_player(self, player: Player):
        """
        Walk forward toward the given player's side or backward if needed.
        FIX: Sends velocity command CONTINUOUSLY every 100ms instead of once.
        """
        print(f"  >> Walking forward (x={WALK_SPEED}) for {MOVE_DURATION}s...")
        await self._stream_velocity(x=WALK_SPEED, y=0.0, z=0.0, duration=MOVE_DURATION)
        self.stop_move()
        print(f"  >> Walk done")

    async def celebrate_win(self, player: Player):
        """Perform victory animation for the winning player.
        Turns toward winner, walks forward, returns to neutral, then dances."""
        print(f"\n{'='*60}")
        print(f"  [TROPHY] PLAYER {player.name} WINS THE GAME! [TROPHY]")
        print(f"{'='*60}\n")

        if not self.connected:
            print("  >> (Dog offline, skipping celebration)")
            return

        # 1. Face the winner (90 deg turn)
        await self.face_player(player)
        await asyncio.sleep(0.2)

        # 2. Walk forward toward the winner (final approach)
        print("  >> Final walk toward winner...")
        await self._stream_velocity(x=WALK_SPEED, y=0.0, z=0.0, duration=MOVE_DURATION)
        self.stop_move()
        await asyncio.sleep(0.2)

        # 3. Return to neutral (90 deg back)
        await self.return_to_neutral()
        await asyncio.sleep(0.3)

        # 4. Dance!
        print("  >> DANCE!")
        self.celebrate()
        try:
            await asyncio.wait_for(asyncio.sleep(4), timeout=5.0)
        except asyncio.TimeoutError:
            print("  >> (Celebration timed out, continuing)")

    async def reset_to_neutral(self):
        """Return the dog to neutral (perpendicular to players) using return_to_neutral."""
        if self._last_turn_direction is not None:
            await self.return_to_neutral()
        else:
            self.facing_angle = 0.0

    async def _keepalive_loop(self):
        """
        Periodically sends a lightweight read-only query (GetBodyHeight) to
        keep the WebRTC data channel alive without causing any robot motion.
        """
        while self.connected and self.dc:
            try:
                if self.dc.readyState == "open":
                    # Use a read-only query that does NOT trigger any motion
                    self._send_cmd(SPORT_CMD["GetBodyHeight"])
            except Exception:
                pass
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break

    def start_keepalive(self):
        """Start the background keepalive task."""
        if self.connected and self.dc and self._keepalive_task is None:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def stop_keepalive(self):
        """Stop the background keepalive task."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    async def disconnect(self):
        """Disconnect from the robot."""
        self.stop_keepalive()
        self.connected = False
        if self.conn:
            await self.conn.disconnect()
            print("  Disconnected.")


# =============================================================================
# MAIN GAME
# =============================================================================

def draw_ui(frame, state: GameState, side_assign: SideAssignment,
            p1_result, p2_result, robot_connected=True):
    """
    Draw the game UI on the frame for the webcam display.
    Player labels are positioned according to the side assignment.
    """
    h, w, _ = frame.shape

    # Vertical divider line (always splits left/right halves)
    mid_x = w // 2
    cv2.line(frame, (mid_x, 0), (mid_x, h), (100, 100, 100), 2)

    # Player A label — on its assigned side
    a_side = side_assign.get_side_for_player(Player.A)
    if a_side == Side.LEFT:
        a_label_x = 10
        b_label_x = mid_x + 10
    else:
        a_label_x = mid_x + 10
        b_label_x = 10

    cv2.putText(frame, "PLAYER A", (a_label_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    cv2.putText(frame, f"{side_assign.side_name(Player.A)} half", (a_label_x, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

    # Player B label — on its assigned side
    cv2.putText(frame, "PLAYER B", (b_label_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
    cv2.putText(frame, f"{side_assign.side_name(Player.B)} half", (b_label_x, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Player A result (on its own side)
    p1_x = a_label_x if side_assign.get_side_for_player(Player.A) == Side.LEFT else b_label_x
    if p1_result:
        gesture, score, _ = p1_result
        g_text = gesture if gesture else "detecting..."
        cv2.putText(frame, f"Gesture: {g_text}", (p1_x, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        cv2.putText(frame, f"Accuracy: {score:.1f}", (p1_x, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    # Player B result (on its own side)
    if side_assign.get_side_for_player(Player.B) == Side.LEFT:
        p2_x = a_label_x
    else:
        p2_x = b_label_x
    if p2_result:
        gesture, score, _ = p2_result
        g_text = gesture if gesture else "detecting..."
        cv2.putText(frame, f"Gesture: {g_text}", (p2_x, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        cv2.putText(frame, f"Accuracy: {score:.1f}", (p2_x, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Scale bar at the bottom
    bar_y = h - 60
    bar_x = 50
    bar_w = w - 100
    bar_h = 30

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    # Colored sections
    a_color = (0, 200, 80)
    b_color = (80, 150, 255)
    neutral_color = (100, 100, 100)

    # Draw the scale bar with gradient
    for i in range(-WIN_SCORE, WIN_SCORE + 1):
        pos = bar_x + int((i + WIN_SCORE) / (2 * WIN_SCORE) * bar_w)
        cv2.line(frame, (pos, bar_y), (pos, bar_y + bar_h),
                 (120, 120, 120), 1)

    # Current score indicator
    # Player A score fills to the LEFT of center (Player A's side)
    # Player B score fills to the RIGHT of center (Player B's side)
    mid_pos = bar_x + bar_w // 2
    if state.scale >= 0:
        bar_fill = int((state.scale / WIN_SCORE) * (bar_w / 2))
        cv2.rectangle(frame, (mid_pos - bar_fill, bar_y), (mid_pos, bar_y + bar_h),
                      a_color, -1)
    else:
        bar_fill = int((-state.scale / WIN_SCORE) * (bar_w / 2))
        cv2.rectangle(frame, (mid_pos, bar_y), (mid_pos + bar_fill, bar_y + bar_h),
                      b_color, -1)

    # Center mark
    cv2.line(frame, (bar_x + bar_w // 2, bar_y),
             (bar_x + bar_w // 2, bar_y + bar_h), (200, 200, 200), 2)

    # Labels on scale
    cv2.putText(frame, f"Player B {abs(state.scale):d}", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, b_color, 1)
    cv2.putText(frame, f"Player A {state.scale:d}", (bar_x + bar_w - 80, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, a_color, 1)
    cv2.putText(frame, f"-{WIN_SCORE} (B)", (bar_x, bar_y + bar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, b_color, 1)
    cv2.putText(frame, f"+{WIN_SCORE} (A)", (bar_x + bar_w - 55, bar_y + bar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, a_color, 1)

    # Current scale value
    val_text = f"SCORE: {state.scale:+d}"
    cv2.putText(frame, val_text, (bar_x + bar_w // 2 - 50, bar_y + bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Game over overlay
    if state.game_over:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        winner_name = f"PLAYER {state.winner.name}"
        win_text = f"{winner_name} WINS THE GAME!"
        (tw, th), _ = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        tx = (w - tw) // 2
        ty = h // 2 - 20
        cv2.putText(frame, win_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 255) if state.winner == Player.A else (100, 200, 255),
                    3)

        cv2.putText(frame, "Press 'r' to restart, 'q' to quit",
                    (w // 2 - 150, ty + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Round info
    cv2.putText(frame, f"Round: {state.round}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Dog connection status
    dog_status = "Dog: [ON] Connected" if robot_connected else "Dog: [OFF] Offline"
    dog_color = (0, 255, 0) if robot_connected else (0, 0, 255)
    cv2.putText(frame, dog_status, (w // 2 - 80, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dog_color, 1)

    cv2.putText(frame, "q=quit  r=restart", (w - 160, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def _select_single_camera(available, player_name, timeout=15, allow_same_as=None):
    """
    Interactive camera selection: shows a preview window and lets the user
    pick a camera by pressing the number key. Returns the selected camera index,
    or None if the user quits.
    """
    import cv2 as _cv2
    cur_cam = available[0]
    menu_active = True
    start_time = time.time()

    preview_cam = _cv2.VideoCapture(cur_cam, _cv2.CAP_DSHOW)
    if preview_cam.isOpened():
        preview_cam.set(_cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        preview_cam.set(_cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while menu_active:
        if preview_cam is None or not preview_cam.isOpened():
            preview_cam = _cv2.VideoCapture(cur_cam, _cv2.CAP_DSHOW)
            if preview_cam.isOpened():
                preview_cam.set(_cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                preview_cam.set(_cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if preview_cam and preview_cam.isOpened():
            ret, frame = preview_cam.read()
            if ret:
                frame = _cv2.flip(frame, 1)
                h, w, _ = frame.shape
                overlay = frame.copy()
                _cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                _cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                _cv2.putText(frame, f"SELECT CAMERA FOR {player_name.upper()}", (w//2 - 200, 50),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                _cv2.putText(frame, f"Selected: Camera {cur_cam}", (w//2 - 120, 100),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                y = 150
                for idx in available:
                    color = (0, 255, 0) if idx == cur_cam else (180, 180, 180)
                    marker = " <<" if idx == cur_cam else ""
                    _cv2.putText(frame, f"  [{idx}] Camera {idx}{marker}", (w//2 - 120, y),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y += 35

                if allow_same_as is not None:
                    _cv2.putText(frame, "[s] Same as Player A  |  [Space/Enter] Confirm  |  [q] Quit",
                                 (w//2 - 260, h - 50), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                else:
                    _cv2.putText(frame, "[Space/Enter] Confirm  |  [q] Quit",
                                 (w//2 - 180, h - 50), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

                _cv2.imshow("Better Commander - Camera Setup", frame)

        key = _cv2.waitKey(30) & 0xFF
        key_char = chr(key) if 32 <= key < 256 else ""

        if key_char.isdigit() and int(key_char) in available:
            cur_cam = int(key_char)
            print(f"  >> {player_name} -> Camera {cur_cam}")
            if preview_cam:
                preview_cam.release()
                preview_cam = None
        elif key == ord('s') and allow_same_as is not None and cur_cam != allow_same_as:
            cur_cam = allow_same_as
            print(f"  >> {player_name} -> same as Player A (Camera {cur_cam})")
            if preview_cam:
                preview_cam.release()
                preview_cam = None
        elif key == ord(' ') or key == 13:
            menu_active = False
        elif time.time() - start_time > timeout:
            print(f"  >> Auto-selected Camera {cur_cam} for {player_name} after {timeout}s")
            menu_active = False
        elif key == ord('q'):
            if preview_cam:
                preview_cam.release()
            _cv2.destroyAllWindows()
            return None

    if preview_cam:
        preview_cam.release()
    return cur_cam


async def main():
    print("=" * 60)
    print("  Better Commander — Go2 Air Game")
    print("  Two-Player Hand Gesture Accuracy Duel")
    print("=" * 60)
    print(f"\n  Win at score ±{WIN_SCORE}")
    print(f"  Robot IP: {ROBOT_IP}")
    print(f"\n  HOW TO PLAY:")
    print(f"  1. Both players show a hand gesture to the camera")
    print(f"  2. The system judges which gesture is MORE ACCURATE")
    print(f"  3. More accurate player scores +1 point")
    print(f"  4. Robot dog walks toward the scoring player")
    print(f"  5. Robot dog faces the scoring player")
    print(f"  6. Score ±{WIN_SCORE} to WIN!")
    print(f"\n  Controls:")
    print(f"    'q' = quit")
    print(f"    'r' = restart")
    print()

    print("\n  >> Starting webcams...")

    # --- Initialize gesture scorer ---
    scorer = GestureScorer()

    # --- DUAL CAMERA SELECTION MENU ---
    # Scan for available cameras (indices 0-9)
    print("  Scanning for cameras...")
    available = []
    for i in range(10):
        test_cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if test_cam.isOpened():
            ok, _ = test_cam.read()
            if ok:
                available.append(i)
                print(f"    Camera {i}: [AVAILABLE]")
            test_cam.release()
        else:
            test_cam.release()

    if not available:
        print("  No cameras found!")
        return

    # Phase 1: Select Camera for Player A
    print("\n" + "=" * 60)
    print("  SELECT CAMERA FOR PLAYER A")
    print("=" * 60)
    cam_a = _select_single_camera(available, "Player A", 15)
    if cam_a is None:
        return

    # Phase 2: Select Camera for Player B
    print("\n" + "=" * 60)
    print("  SELECT CAMERA FOR PLAYER B")
    print("=" * 60)
    # Show Player B options including "same as Player A" option
    cam_b = _select_single_camera(available, "Player B", 15, allow_same_as=cam_a)
    if cam_b is None:
        return

    cam_setup = CameraSetup(cam_a, cam_b)
    print(f"\n  Camera setup: {cam_setup.describe()}")

    # --- Open camera(s) ---
    same_cam = (cam_a == cam_b)
    cam_a_cap = cv2.VideoCapture(cam_a, cv2.CAP_DSHOW)
    if not cam_a_cap.isOpened():
        print(f"  Could not open Camera {cam_a}!")
        return
    cam_a_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam_a_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if same_cam:
        cam_b_cap = cam_a_cap  # same object, no need for second capture
        print(f"  Camera {cam_a} ready (split-screen for both players)")
    else:
        cam_b_cap = cv2.VideoCapture(cam_b, cv2.CAP_DSHOW)
        if not cam_b_cap.isOpened():
            print(f"  Could not open Camera {cam_b}!")
            cam_a_cap.release()
            return
        cam_b_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cam_b_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"  Camera {cam_a} (Player A) and Camera {cam_b} (Player B) ready!")

    # --- Side assignment: decide which side each player appears on in composite ---
    side_assign = SideAssignment()  # default: A=left, B=right
    menu_active = True
    menu_start_time = time.time()
    print("\n" + "=" * 60)
    print("  SIDE SELECTION MENU")
    print("=" * 60)
    print("  Decide which player is on the LEFT vs RIGHT in the combined view.")
    print(f"  Current: {side_assign.describe()}")
    print()
    print("  Keys:")
    print("    's'  = Swap sides (A <-> B)")
    print("    ' ' (space) / Enter = Confirm & start game")
    print("  (The menu auto-confirms after 20 seconds)")
    print()

    # Determine if same camera or two cameras

    while menu_active:
        ret_a, frame_a = cam_a_cap.read()
        if same_cam:
            ret_b, frame_b = ret_a, frame_a
        else:
            ret_b, frame_b = cam_b_cap.read()
        if ret_a and ret_b:
            frame_a = cv2.flip(frame_a, 1)
            if not same_cam:
                frame_b = cv2.flip(frame_b, 1)

            if same_cam:
                # Single camera → just split the frame left/right
                h, w, _ = frame_a.shape
                mid_x = w // 2
                composite = frame_a.copy()
            else:
                # Two cameras → stitch side by side
                h = min(frame_a.shape[0], frame_b.shape[0])
                frame_a = cv2.resize(frame_a, (FRAME_WIDTH, h))
                frame_b = cv2.resize(frame_b, (FRAME_WIDTH, h))
                composite = np.hstack((frame_a, frame_b))

            ch, cw, _ = composite.shape

            # Overlay
            overlay = composite.copy()
            cv2.rectangle(overlay, (0, 0), (cw, ch), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, composite, 0.5, 0, composite)

            mid_x = cw // 2
            cv2.line(composite, (mid_x, 0), (mid_x, ch), (100, 100, 100), 2)

            cv2.putText(composite, "CAMERA LAYOUT", (cw//2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(composite, side_assign.describe(), (cw//2 - 200, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Camera labels
            if same_cam:
                label_a = "Split-screen"
                label_b = "Split-screen"
            else:
                label_a = f"Cam {cam_a}"
                label_b = f"Cam {cam_b}"
            cv2.putText(composite, label_a, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 180), 1)
            cv2.putText(composite, label_b, (mid_x + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 180, 180), 1)

            # Which player on which side
            left_player = side_assign.get_player_for_side(Side.LEFT)
            right_player = side_assign.get_player_for_side(Side.RIGHT)
            cv2.putText(composite, f"<- {left_player.name}", (10, ch//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 100) if left_player == Player.A else (100, 200, 255), 2)
            cv2.putText(composite, f"{right_player.name} ->", (cw - 160, ch//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 100) if right_player == Player.A else (100, 200, 255), 2)

            cv2.putText(composite, "[s] Swap  |  [Space/Enter] Start", (cw//2 - 170, ch - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Better Commander - Layout Menu", composite)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            side_assign = SideAssignment(
                a_side=side_assign.b_side,
                b_side=side_assign.a_side
            )
            print(f"  >> Swapped! Now: {side_assign.describe()}")
        elif key == ord(' ') or key == 13:
            menu_active = False
            print(f"  >> Confirmed: {side_assign.describe()}")
        elif time.time() - menu_start_time > 20:
            print(f"  >> Auto-confirmed after 20s: {side_assign.describe()}")
            menu_active = False
        elif key == ord('q'):
            cam_a_cap.release()
            cam_b_cap.release()
            cv2.destroyAllWindows()
            print("  Quit from menu.")
            return

    cv2.destroyWindow("Better Commander - Layout Menu")

    print(f"\n{'='*60}")
    print("  GAME START!")
    print(f"{'='*60}\n")

    # --- Connect to robot (AFTER camera setup to avoid timeout) ---
    robot = RobotController()
    conn_ok = await robot.connect()
    print(f"  >> connect() returned: {conn_ok}, robot.connected = {robot.connected}")

    if robot.connected:
        await asyncio.sleep(0.5)
        print("  >> Standing up...")
        robot.stand_up()
        await asyncio.sleep(2)
        print("  >> Dog should now be standing.")

        print("  >> Sending Hello as connectivity test...")
        robot._send_cmd(SPORT_CMD["Hello"])
        await asyncio.sleep(2)
        print("  >> Hello done.")

    # --- Game loop ---
    state = GameState()
    running = True
    dog_offline_reported = False

    try:
        while running:
            ret_a, frame_a = cam_a_cap.read()
            if same_cam:
                ret_b, frame_b = ret_a, frame_a
            else:
                ret_b, frame_b = cam_b_cap.read()
            if not ret_a or not ret_b:
                await asyncio.sleep(0.03)
                continue

            frame_a = cv2.flip(frame_a, 1)
            if same_cam:
                # Same camera → single frame, split left/right
                h, w, _ = frame_a.shape
                mid_x = w // 2
                composite = frame_a.copy()
                # Draw split line
                cv2.line(composite, (mid_x, 0), (mid_x, h), (100, 100, 100), 2)
                ch, cw, _ = composite.shape
            else:
                # Two cameras → stitch side by side
                frame_b = cv2.flip(frame_b, 1)
                h = min(frame_a.shape[0], frame_b.shape[0])
                frame_a = cv2.resize(frame_a, (FRAME_WIDTH, h))
                frame_b = cv2.resize(frame_b, (FRAME_WIDTH, h))
                composite = np.hstack((frame_a, frame_b))
                ch, cw, _ = composite.shape

            # --- Detect both players on the composite using side assignment ---
            a_roi = side_assign.get_roi(Player.A, cw)
            b_roi = side_assign.get_roi(Player.B, cw)
            # Single MediaPipe pass for both players (much faster than two passes)
            p1_result, p2_result = scorer.detect_both(composite, a_roi, b_roi)

            p1_gesture, p1_score, p1_lm = p1_result
            p2_gesture, p2_score, p2_lm = p2_result

            # --- Track whether both players have a gesture ---
            both_detected = p1_gesture is not None and p2_gesture is not None

            # --- Report dog offline once per transition ---
            if robot.connected is False and not dog_offline_reported:
                print("  >> [WARN] Dog communication lost. Game continues without robot movement.")
                dog_offline_reported = True
            elif robot.connected:
                dog_offline_reported = False

            # --- Annotate landmarks on composite ---
            conns = [
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
            ]

            if p1_lm:
                for i, pt in enumerate(p1_lm):
                    x, y = int(pt.x * cw), int(pt.y * ch)
                    cv2.circle(composite, (x, y), 5, (0, 255, 100), -1)
                for i, j in conns:
                    cv2.line(composite,
                             (int(p1_lm[i].x * cw), int(p1_lm[i].y * ch)),
                             (int(p1_lm[j].x * cw), int(p1_lm[j].y * ch)),
                             (0, 255, 100), 2)

            if p2_lm:
                for i, pt in enumerate(p2_lm):
                    x, y = int(pt.x * cw), int(pt.y * ch)
                    cv2.circle(composite, (x, y), 5, (100, 200, 255), -1)
                for i, j in conns:
                    cv2.line(composite,
                             (int(p2_lm[i].x * cw), int(p2_lm[i].y * ch)),
                             (int(p2_lm[j].x * cw), int(p2_lm[j].y * ch)),
                             (100, 200, 255), 2)

            # --- Evaluate round (prep phase + judge) ---
            now = time.time()
            if not state.game_over and not state.is_waiting:
                if not state.prep_first_gesture_seen:
                    if p1_gesture is not None or p2_gesture is not None:
                        state.prep_start_time = now
                        state.prep_first_gesture_seen = True
                        print(f"  >> First gesture detected! Prep phase: {PREP_TIME:.0f}s countdown started...")

                elapsed = now - state.prep_start_time

                if elapsed >= PREP_TIME and both_detected and now - state.last_round_time > GESTURE_COOLDOWN:
                    state.last_round_time = now
                    state.prep_start_time = 0
                    state.prep_first_gesture_seen = False
                    state.prep_gestures_detected = False

                    if p1_score > p2_score:
                        winner = Player.A
                        print(f"  ROUND {state.round+1}: Player A wins! "
                              f"(A:{p1_score:.1f} > B:{p2_score:.1f}) "
                              f"Gesture A={p1_gesture} vs B={p2_gesture}")
                    elif p2_score > p1_score:
                        winner = Player.B
                        print(f"  ROUND {state.round+1}: Player B wins! "
                              f"(B:{p2_score:.1f} > A:{p1_score:.1f}) "
                              f"Gesture A={p1_gesture} vs B={p2_gesture}")
                    else:
                        # Tie — no one scores
                        print(f"  ROUND {state.round+1}: Tie! Both players equal "
                              f"(A:{p1_score:.1f} B:{p2_score:.1f})")
                        winner = None

                    if winner is not None:
                        state.score_point(winner)
                        print(f"  SCALE: {state.scale:+d}  |  {state.get_game_status_text()}")

                        # Robot actions for the scoring player
                        if robot.connected:
                            print(f"  >> Dog acts for Player {winner.name}")
                            await robot.face_player(winner)
                            await asyncio.sleep(0.1)
                            await robot.walk_toward_player(winner)
                            await asyncio.sleep(0.1)
                            await robot.return_to_neutral()
                            await asyncio.sleep(0.1)

                    if state.game_over:
                        print(f"\n{'='*60}")
                        print(f"  [WIN] PLAYER {state.winner.name} WINS! [WIN]")
                        print(f"{'='*60}\n")
                        if robot.connected:
                            await robot.celebrate_win(state.winner)

                elif elapsed >= PREP_TIME and not both_detected and state.prep_start_time > 0:
                    if state.prep_gestures_detected != "waiting":
                        print("  >> Prep time done — waiting for both players to show gestures...")
                        state.prep_gestures_detected = "waiting"

            # --- Draw UI on composite ---
            draw_ui(composite, state, side_assign, p1_result, p2_result, robot.connected)

            # --- Show "Get ready!" timer when waiting between rounds ---
            if state.is_waiting and not state.game_over:
                remaining = max(0, ROUND_WAIT_TIME - (time.time() - state.last_round_time))
                wait_text = f"Next round in {remaining:.0f}s — Get ready!"
                (tw, th), _ = cv2.getTextSize(wait_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                wx = (cw - tw) // 2
                wy = ch - 100
                cv2.rectangle(composite,
                              (wx - 10, wy - th - 10),
                              (wx + tw + 10, wy + 10),
                              (0, 0, 0), -1)
                cv2.putText(composite, wait_text, (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # --- Show "Preparing..." countdown during prep phase ---
            if state.prep_start_time > 0 and state.prep_first_gesture_seen:
                remaining = max(0, PREP_TIME - (time.time() - state.prep_start_time))
                if both_detected:
                    prep_text = f"Round {state.round+1} — Good! Locking in {remaining:.0f}s"
                else:
                    prep_text = f"Round {state.round+1} — Show both gestures! Locking in {remaining:.0f}s"
                (tw, th), _ = cv2.getTextSize(prep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                wx = (cw - tw) // 2
                wy = ch - 100  # Just above the scale bar
                cv2.rectangle(composite,
                              (wx - 10, wy - th - 10),
                              (wx + tw + 10, wy + 10),
                              (0, 0, 0), -1)
                cv2.putText(composite, prep_text, (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # --- Show "Waiting for both players" after countdown expired ---
            if (state.prep_start_time > 0 and state.prep_first_gesture_seen
                    and time.time() - state.prep_start_time >= PREP_TIME
                    and not both_detected):
                wait_both_text = "Countdown done! Waiting for BOTH gestures to evaluate..."
                (tw, th), _ = cv2.getTextSize(wait_both_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                wx = (cw - tw) // 2
                wy = ch - 100  # Just above the scale bar
                cv2.rectangle(composite,
                              (wx - 10, wy - th - 10),
                              (wx + tw + 10, wy + 10),
                              (0, 0, 0), -1)
                cv2.putText(composite, wait_both_text, (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # --- Show composite frame ---
            cv2.imshow("Better Commander - Go2 Air Game", composite)

            # --- Handle keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('r'):
                print("\n  >> RESTARTING GAME...\n")
                state = GameState()
                if robot.connected:
                    robot.stop_move()
                    await asyncio.sleep(0.3)
                    await robot.reset_to_neutral()
                    robot.stop_move()
                print("  >> Game reset. New round starting!\n")

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n  Interrupted!")
    finally:
        cam_a_cap.release()
        if cam_b_cap is not cam_a_cap:
            cam_b_cap.release()
        cv2.destroyAllWindows()
        if robot.connected:
            print("  >> Sitting down...")
            robot.sit_down()
            await asyncio.sleep(1)
            await robot.disconnect()
        print("  Game ended. Thanks for playing!")


if __name__ == "__main__":
    asyncio.run(main())
