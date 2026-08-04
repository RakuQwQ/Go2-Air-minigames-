"""
Game 7: Freestyle Mode
=======================
Free control of your Go2 Air with keyboard commands.

Controls:
  W / Up arrow     = Walk forward
  S / Down arrow   = Walk backward
  A / Left arrow   = Turn left
  D / Right arrow  = Turn right
  Space            = Stop moving
  1                = Stand up
  2                = Sit down
  3                = Say hello
  4                = Dance 1
  5                = Dance 2
  6                = Finger heart
  7                = Stretch
  8                = Wiggle hips
  9                = Body tilt right
  0                = Body tilt center
  Q or Esc         = Exit
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import (
    move_cmd, stand_up, sit_down, stop_moving, say_hello,
    dance, finger_heart, stretch, wiggle_hips, euler_cmd,
)


def print_controls():
    """Display the control scheme."""
    print()
    print("  " + "=" * 50)
    print("  FREESTYLE MODE - Controls")
    print("  " + "=" * 50)
    print()
    print("    W/Up    = Forward      S/Down   = Backward")
    print("    A/Left  = Turn Left    D/Right  = Turn Right")
    print("    Space   = Stop")
    print()
    print("    1 = Stand up     2 = Sit down")
    print("    3 = Hello        4 = Dance 1")
    print("    5 = Dance 2      6 = Finger Heart")
    print("    7 = Stretch      8 = Wiggle Hips")
    print("    9 = Tilt Right   0 = Tilt Center")
    print()
    print("    Q = Exit")
    print()
    print("  " + "=" * 50)
    print("  Type a command and press Enter:")
    print()


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||            FREESTYLE MODE               ||")
    print("  ||     Free control of your Go2 Air        ||")
    print("  " + "=" * 56)

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    print_controls()

    loop = asyncio.get_event_loop()

    try:
        while True:
            cmd = await loop.run_in_executor(None, sys.stdin.readline)
            cmd = cmd.strip().lower()

            if cmd in ("q", "quit", "exit", "esc"):
                print("\n  >> Exiting Freestyle Mode...")
                break

            # Movement
            elif cmd in ("w", "\x1b[A"):  # w or Up arrow
                robot.send(move_cmd(x=0.3))
                print("  >> Walking forward")
            elif cmd in ("s", "\x1b[B"):  # s or Down arrow
                robot.send(move_cmd(x=-0.3))
                print("  >> Walking backward")
            elif cmd in ("a", "\x1b[D"):  # a or Left arrow
                robot.send(move_cmd(z=0.5))
                print("  >> Turning left")
            elif cmd in ("d", "\x1b[C"):  # d or Right arrow
                robot.send(move_cmd(z=-0.5))
                print("  >> Turning right")
            elif cmd in ("", " "):
                robot.send(stop_moving())
                print("  >> Stop")

            # Actions
            elif cmd == "1":
                robot.send(stand_up())
                print("  >> Standing up!")
            elif cmd == "2":
                robot.send(sit_down())
                print("  >> Sitting down")
            elif cmd == "3":
                robot.send(say_hello())
                print("  >> Hello!")
            elif cmd == "4":
                robot.send(dance(1))
                print("  >> Dancing! (style 1)")
            elif cmd == "5":
                robot.send(dance(2))
                print("  >> Dancing! (style 2)")
            elif cmd == "6":
                robot.send(finger_heart())
                print("  >> Finger Heart! <3")
            elif cmd == "7":
                robot.send(stretch())
                print("  >> Stretching!")
            elif cmd == "8":
                robot.send(wiggle_hips())
                print("  >> Wiggle wiggle!")
            elif cmd == "9":
                robot.send(euler_cmd(roll=0.4))
                print("  >> Tilting right")
            elif cmd == "0":
                robot.send(euler_cmd(roll=0.0))
                print("  >> Body center")

            else:
                print(f"  Unknown command: '{cmd}'")

    except (EOFError, KeyboardInterrupt):
        print("\n\n  >> Interrupted!")

    # Clean shutdown
    print("  >> Sitting down...")
    robot.send(sit_down())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  >> Disconnected. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
