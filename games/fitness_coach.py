"""
Game 6: Fitness Coach
======================
Your Go2 Air becomes a fitness coach!

Watch and follow along as your robot demonstrates exercise moves.
The robot will perform different exercises with rest breaks in between.

Exercises:
  - Hello Stretch      (warm up)
  - Stretch            (full body stretch)
  - Balance Stand      (core stability)
  - Body Tilt Left     (side stretch)
  - Body Tilt Right    (side stretch)
  - Wiggle Hips        (hip mobility)
  - Dance Cardio       (get your heart rate up!)
  - Sit & Rise         (squat motion)
  - Recovery Stand     (cool down)
  - Finger Heart       (positive affirmation!)

Controls:
  Enter/Space = Next exercise
  1-9         = Jump to specific exercise
  R           = Restart routine
  Q           = Exit
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import (
    stand_up, sit_down, say_hello, stretch,
    wiggle_hips, dance, finger_heart, euler_cmd,
    sport_cmd, stop_moving,
)
from unitree_webrtc_connect import SPORT_CMD


# Fitness routine
EXERCISES = [
    {"name": "Warm Up - Hello!", "cmd": say_hello, "duration": 3,
     "desc": "Wave and say hello! Great for warming up those shoulders."},
    {"name": "Full Body Stretch", "cmd": stretch, "duration": 3,
     "desc": "Stretch those legs! Feel the burn!"},
    {"name": "Core Stability - Balance", "cmd": lambda: sport_cmd(SPORT_CMD["BalanceStand"]), "duration": 4,
     "desc": "Hold that balance! Engage your core!"},
    {"name": "Side Stretch - Left", "cmd": lambda: euler_cmd(roll=-0.3), "duration": 3,
     "desc": "Lean to the left! Feel that side stretch!"},
    {"name": "Side Stretch - Right", "cmd": lambda: euler_cmd(roll=0.3), "duration": 3,
     "desc": "Now lean to the right! Great for obliques!"},
    {"name": "Hip Mobility - Wiggle", "cmd": wiggle_hips, "duration": 3,
     "desc": "Wiggle those hips! Loosen up!"},
    {"name": "Cardio - Dance!", "cmd": lambda: dance(1), "duration": 5,
     "desc": "Get that heart rate up! Dance it out!"},
    {"name": "Squat Motion - Sit & Rise", "cmd": sit_down, "duration": 2,
     "desc": "Sit down... and get back up! Full body workout!"},
    {"name": "Cool Down - Recovery", "cmd": lambda: sport_cmd(SPORT_CMD["RecoveryStand"]), "duration": 3,
     "desc": "Recovery pose. Breathe in... breathe out..."},
    {"name": "Positive Affirmation", "cmd": finger_heart, "duration": 3,
     "desc": "You did great! Here's a heart for you! <3"},
]


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||          FITNESS COACH                 ||")
    print("  ||   Get fit with your Go2 Air!           ||")
    print("  " + "=" * 56)

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    print()
    print("  Follow along as your Go2 demonstrates exercises!")
    print("  Press Enter to start each exercise.")
    print("  Press Q to quit anytime.")
    print()

    input("  >>> Press Enter to start the workout! >>>")

    loop = asyncio.get_event_loop()
    exercise_index = 0

    try:
        while exercise_index < len(EXERCISES):
            exercise = EXERCISES[exercise_index]

            print()
            print("  " + "-" * 50)
            print(f"  Exercise {exercise_index + 1}/{len(EXERCISES)}: {exercise['name']}")
            print(f"  \"{exercise['desc']}\"")
            print("  " + "-" * 50)
            print()

            # Show the exercise
            robot.send(stand_up())
            await asyncio.sleep(1)
            robot.send(exercise["cmd"]())
            await asyncio.sleep(exercise["duration"])

            print("  >>> Exercise done! Rest time.")
            print("  >>> Press Enter for next exercise, or Q to quit.")

            while True:
                cmd = await loop.run_in_executor(None, sys.stdin.readline)
                cmd = cmd.strip().upper()

                if cmd == "Q":
                    raise KeyboardInterrupt()
                elif cmd == "R":
                    exercise_index = 0
                    print("\n  >> Restarting routine!")
                    break
                elif cmd == "" or cmd == " ":
                    exercise_index += 1
                    break
                else:
                    # Try number jump
                    try:
                        num = int(cmd)
                        if 1 <= num <= len(EXERCISES):
                            exercise_index = num - 1
                            break
                    except ValueError:
                        pass
                    print("  Press Enter for next, Q to quit.")

    except (EOFError, KeyboardInterrupt):
        print("\n\n  >> Workout interrupted!")

    # Clean shutdown
    print("\n  >> Cool down...")
    robot.send(sit_down())
    await asyncio.sleep(2)
    robot.send(finger_heart())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  >> Great workout! You earned that heart! <3\n")


if __name__ == "__main__":
    asyncio.run(main())
