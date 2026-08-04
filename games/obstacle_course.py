"""
Game 3: Obstacle Course
========================
Navigate your Go2 Air through an obstacle course!

Guide the robot around objects and through a course using precise controls.
You can set waypoints and time yourself.

Controls:
  W          = Walk forward
  S          = Walk backward
  A          = Turn left
  D          = Turn right
  Space      = Stop
  P          = Place waypoint marker
  T          = Show course time
  R          = Reset to start position
  H          = Show help
  Q          = Exit

Tip: Place objects (chairs, boxes) and try to navigate around them!
"""

import asyncio
import sys
import os
import time
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import (
    move_cmd, stand_up, sit_down, stop_moving,
    euler_cmd, sport_cmd,
)
from unitree_webrtc_connect import SPORT_CMD


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||          OBSTACLE COURSE               ||")
    print("  ||    Navigate your Go2 through objects!  ||")
    print("  " + "=" * 56)

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    # Start
    robot.send(stand_up())
    await asyncio.sleep(2)

    print()
    print("  " + "=" * 50)
    print("  Controls:")
    print("    W = Forward        S = Backward")
    print("    A = Turn Left      D = Turn Right")
    print("    Space = Stop       P = Place waypoint")
    print("    T = Course time    R = Reset position")
    print("    H = Help           Q = Exit")
    print("  " + "=" * 50)
    print()
    print("  >>> Place some objects around and navigate through them!")
    print("  >>> Type T to see your course time.")
    print()

    start_time = time.time()
    waypoints = 0
    loop = asyncio.get_event_loop()

    try:
        while True:
            cmd = await loop.run_in_executor(None, sys.stdin.readline)
            cmd = cmd.strip().lower()

            if cmd == "q":
                elapsed = time.time() - start_time
                print(f"\n  >> Course finished in {timedelta(seconds=int(elapsed))}")
                print(f"  >> Waypoints placed: {waypoints}")
                break

            elif cmd == "w":
                robot.send(move_cmd(x=0.3))
                print("  >> Forward (release with Space)")
            elif cmd == "s":
                robot.send(move_cmd(x=-0.3))
                print("  >> Backward (release with Space)")
            elif cmd == "a":
                robot.send(move_cmd(z=0.5))
                print("  >> Turning left")
            elif cmd == "d":
                robot.send(move_cmd(z=-0.5))
                print("  >> Turning right")
            elif cmd in ("", " "):
                robot.send(stop_moving())
                print("  >> Stopped")
            elif cmd == "p":
                waypoints += 1
                print(f"  >> Waypoint {waypoints} placed!")
            elif cmd == "t":
                elapsed = time.time() - start_time
                print(f"  >> Course time: {timedelta(seconds=int(elapsed))}")
            elif cmd == "r":
                print("  >> Resetting... (balance stand)")
                robot.send(sport_cmd(SPORT_CMD["BalanceStand"]))
                await asyncio.sleep(1)
            elif cmd == "h":
                print()
                print("  Controls:")
                print("    W = Forward        S = Backward")
                print("    A = Turn Left      D = Turn Right")
                print("    Space = Stop       P = Place waypoint")
                print("    T = Course time    R = Reset position")
                print("    H = Help           Q = Exit")
                print()

    except (EOFError, KeyboardInterrupt):
        elapsed = time.time() - start_time
        print(f"\n\n  >> Course time: {timedelta(seconds=int(elapsed))}")
        print(f"  >> Waypoints: {waypoints}")

    # Clean shutdown
    print("  >> Sitting down...")
    robot.send(sit_down())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  >> Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
