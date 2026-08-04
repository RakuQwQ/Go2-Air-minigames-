"""
Game 1: Dance Commander
========================
Choreograph dance routines for your Go2 Air!

Create a sequence of moves and watch your Go2 perform a dance routine.

Controls:
  1-8    = Add a move to your routine
  P      = Play the routine
  C      = Clear the routine
  L      = List current routine
  S      = Save routine to file
  Q      = Exit

Commands:
  [1] Stand Up    [2] Sit Down    [3] Hello
  [4] Dance 1     [5] Dance 2     [6] Finger Heart
  [7] Stretch     [8] Wiggle Hips [9] Body Tilt
  [0] Recovery Stand
"""

import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import (
    sport_cmd, move_cmd, euler_cmd,
    stand_up, sit_down, say_hello, dance,
    finger_heart, stretch, wiggle_hips,
)
from unitree_webrtc_connect import SPORT_CMD


# Available dance moves
DANCE_MOVES = {
    "1": {"name": "Stand Up", "cmd": stand_up, "duration": 2},
    "2": {"name": "Sit Down", "cmd": sit_down, "duration": 2},
    "3": {"name": "Hello", "cmd": say_hello, "duration": 3},
    "4": {"name": "Dance 1", "cmd": lambda: dance(1), "duration": 4},
    "5": {"name": "Dance 2", "cmd": lambda: dance(2), "duration": 4},
    "6": {"name": "Finger Heart", "cmd": finger_heart, "duration": 3},
    "7": {"name": "Stretch", "cmd": stretch, "duration": 3},
    "8": {"name": "Wiggle Hips", "cmd": wiggle_hips, "duration": 3},
    "9": {"name": "Tilt Right", "cmd": lambda: euler_cmd(roll=0.4), "duration": 2},
    "0": {"name": "Recovery Stand", "cmd": lambda: sport_cmd(SPORT_CMD["RecoveryStand"]), "duration": 3},
}


def print_menu():
    """Print the dance commander menu."""
    print()
    print("  " + "=" * 50)
    print("  DANCE COMMANDER - Choreograph your Go2!")
    print("  " + "=" * 50)
    print()
    print("  Available moves:")
    for key, move in DANCE_MOVES.items():
        print(f"    [{key}] {move['name']:15s} ({move['duration']}s)")
    print()
    print("  Commands:")
    print("    [P] Play routine    [C] Clear routine")
    print("    [L] List routine    [S] Save routine")
    print("    [Q] Exit")
    print()


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||           DANCE COMMANDER              ||")
    print("  ||    Choreograph your Go2's dance moves  ||")
    print("  " + "=" * 56)

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    routine = []
    loop = asyncio.get_event_loop()

    print_menu()

    try:
        while True:
            cmd = await loop.run_in_executor(None, sys.stdin.readline)
            cmd = cmd.strip().upper()

            if cmd == "Q":
                print("\n  >> Exiting Dance Commander...")
                break

            elif cmd == "P":
                if not routine:
                    print("\n  >> Your routine is empty! Add some moves first.")
                    continue
                print(f"\n  >> Playing routine ({len(routine)} moves)...")
                robot.send(stand_up())
                await asyncio.sleep(2)
                for i, move_key in enumerate(routine):
                    move = DANCE_MOVES[move_key]
                    print(f"  >> [{i+1}/{len(routine)}] {move['name']}")
                    robot.send(move["cmd"]())
                    await asyncio.sleep(move["duration"])
                print("\n  >> Routine complete!")
                robot.send(sit_down())

            elif cmd == "C":
                routine = []
                print("\n  >> Routine cleared!")

            elif cmd == "L":
                if not routine:
                    print("\n  >> Routine is empty.")
                else:
                    print(f"\n  >> Current routine ({len(routine)} moves):")
                    for i, move_key in enumerate(routine):
                        move = DANCE_MOVES[move_key]
                        print(f"     {i+1}. {move['name']} ({move['duration']}s)")

            elif cmd == "S":
                if not routine:
                    print("\n  >> Nothing to save.")
                    continue
                filename = f"dance_routine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w") as f:
                    json.dump(routine, f)
                print(f"\n  >> Routine saved to {filename}")

            elif cmd in DANCE_MOVES:
                move = DANCE_MOVES[cmd]
                routine.append(cmd)
                print(f"\n  >> Added: {move['name']} ({len(routine)} moves in routine)")
                # Preview the move on the robot
                robot.send(move["cmd"]())
                await asyncio.sleep(1)

            else:
                print(f"\n  Unknown command: '{cmd}'")

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
