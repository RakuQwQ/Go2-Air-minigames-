"""
Game 4: Gesture Control
========================
Control your Go2 Air with hand gestures using your webcam!

Requires:
  - numpy
  - opencv-python (cv2)
  - mediapipe

Install with:
  pip install numpy opencv-python mediapipe

Gestures:
  Open Palm (5 fingers)  = Walk forward
  Fist (0 fingers)       = Stop
  1 Finger (index up)    = Say Hello / Wave
  2 Fingers (peace)      = Dance!
  3 Fingers              = Sit down
  4 Fingers              = Stand up
  Thumbs Up              = Finger Heart

Note: Gesture recognition requires a working webcam.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import (
    move_cmd, stand_up, sit_down, stop_moving,
    say_hello, dance, finger_heart,
)


# Check for optional dependencies
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    HAS_VISION = True
except ImportError:
    HAS_VISION = False


def get_gesture_name(finger_count, thumb_up):
    """Convert finger count to a gesture name."""
    if thumb_up and finger_count == 0:
        return "thumbs_up"
    if finger_count == 0:
        return "fist"
    if finger_count == 1:
        return "one"
    if finger_count == 2:
        return "peace"
    if finger_count == 3:
        return "three"
    if finger_count == 4:
        return "four"
    if finger_count == 5:
        return "palm"
    return "unknown"


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||          GESTURE CONTROL               ||")
    print("  ||   Control your Go2 with hand gestures! ||")
    print("  " + "=" * 56)

    if not HAS_VISION:
        print()
        print("  This game requires additional packages:")
        print("    pip install numpy opencv-python mediapipe")
        print()
        input("  Press Enter to return to menu...")
        return

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n  ERROR: Could not open webcam!")
        await robot.disconnect()
        return

    print("\n  Gesture Recognition active!")
    print("  Gestures:")
    print("    Palm  (5 fingers) = Walk forward")
    print("    Fist  (0 fingers) = Stop")
    print("    1 Finger          = Hello")
    print("    2 Fingers (peace) = Dance!")
    print("    3 Fingers         = Sit down")
    print("    4 Fingers         = Stand up")
    print("    Thumbs Up         = Finger Heart")
    print()
    print("  Press 'Q' in the camera window to exit.")
    print()

    last_gesture = ""
    last_command_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  >> Camera error!")
                break

            # Flip horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture_display = "No hand detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Count extended fingers
                    h, w, _ = frame.shape
                    landmarks = hand_landmarks.landmark

                    # Tips and PIPs for finger counting
                    finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
                    finger_pips = [3, 6, 10, 14, 18]

                    fingers = []
                    # Thumb
                    if landmarks[4].x < landmarks[3].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other fingers
                    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
                        if landmarks[tip].y < landmarks[pip].y:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    finger_count = sum(fingers)
                    thumb_up = fingers[0] == 1
                    gesture = get_gesture_name(finger_count, thumb_up)

                    # Show gesture
                    gesture_display = f"Gesture: {gesture} ({finger_count} fingers)"

                    # Execute command (with cooldown)
                    current_time = asyncio.get_event_loop().time()
                    if gesture != last_gesture or (current_time - last_command_time) > 2:
                        if gesture == "palm":
                            robot.send(move_cmd(x=0.3))
                            print("  >> Walking forward")
                        elif gesture == "fist":
                            robot.send(stop_moving())
                            print("  >> Stop")
                        elif gesture == "one":
                            robot.send(say_hello())
                            print("  >> Hello!")
                        elif gesture == "peace":
                            robot.send(dance(1))
                            print("  >> Dance!")
                        elif gesture == "three":
                            robot.send(sit_down())
                            print("  >> Sit down")
                        elif gesture == "four":
                            robot.send(stand_up())
                            print("  >> Stand up")
                        elif gesture == "thumbs_up":
                            robot.send(finger_heart())
                            print("  >> Finger Heart! <3")

                        last_gesture = gesture
                        last_command_time = current_time

            # Display info on frame
            cv2.putText(frame, gesture_display, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Go2 Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.05)  # Small yield

    except KeyboardInterrupt:
        print("\n\n  >> Interrupted!")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    # Clean shutdown
    print("  >> Sitting down...")
    robot.send(sit_down())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  >> Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
