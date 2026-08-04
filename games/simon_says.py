"""
Game 2: Simon Says
===================
A memory game for your Go2 Air!

Watch the robot perform a sequence of moves, then repeat them back.
Each round adds one more move to the sequence. How many can you remember?

Moves used in this game:
  - Hello (wave)
  - Dance 1
  - Finger Heart
  - Stretch
  - Wiggle Hips
  - Tilt Right
  - Tilt Left

Controls:
  [1-7]  = Select the move you think comes next
  [Q]    = Quit game
"""

import asyncio
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import (
    say_hello, dance, finger_heart, stretch,
    wiggle_hips, euler_cmd, stand_up, sit_down,
)


# The moves used in Simon Says
SIMON_MOVES = [
    {"key": "1", "name": "Hello", "cmd": say_hello, "duration": 2},
    {"key": "2", "name": "Dance", "cmd": lambda: dance(1), "duration": 3},
    {"key": "3", "name": "Heart", "cmd": finger_heart, "duration": 2},
    {"key": "4", "name": "Stretch", "cmd": stretch, "duration": 2},
    {"key": "5", "name": "Wiggle", "cmd": wiggle_hips, "duration": 2},
    {"key": "6", "name": "Tilt Right", "cmd": lambda: euler_cmd(roll=0.4), "duration": 2},
    {"key": "7", "name": "Tilt Left", "cmd": lambda: euler_cmd(roll=-0.4), "duration": 2},
]

MOVE_NAMES = {m["key"]: m["name"] for m in SIMON_MOVES}


def print_header(score: int, streak: list):
    """Display current game state."""
    print()
    print("  " + "=" * 50)
    print(f"  SIMON SAYS - Round {len(streak) + 1}")
    print("  " + "=" * 50)
    print(f"  Score: {score}")
    print()
    if streak:
        print("  Current sequence so far:")
        seq_str = " -> ".join(MOVE_NAMES[k] for k in streak)
        print(f"    {seq_str}")
    print()


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||            SIMON SAYS                  ||")
    print("  ||  Can you remember the dance pattern?   ||")
    print("  " + "=" * 56)

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    # Start standing
    robot.send(stand_up())
    await asyncio.sleep(2)

    print()
    print("  Watch the robot! It will show you a sequence of moves.")
    print("  Then repeat the sequence by typing the numbers.")
    print()
    print("  Moves:")
    for m in SIMON_MOVES:
        print(f"    [{m['key']}] {m['name']}")
    print()
    print("  When ready, press Enter to start...")
    input()

    score = 0
    streak = []  # The sequence to remember
    loop = asyncio.get_event_loop()

    try:
        while True:
            # Add one more move to the sequence
            new_move = random.choice(SIMON_MOVES)
            streak.append(new_move["key"])

            print_header(score, streak)
            print(f"  >>> Watch the robot perform {len(streak)} moves...")
            await asyncio.sleep(1)

            # Robot shows the sequence
            for mk in streak:
                move = next(m for m in SIMON_MOVES if m["key"] == mk)
                print(f"  >> Robot does: {move['name']}")
                robot.send(move["cmd"]())
                await asyncio.sleep(move["duration"])

            print()
            print("  >>> Your turn! Repeat the sequence:")
            print("  >>> Type each move number and press Enter.")

            # Player repeats
            for i in range(len(streak)):
                expected = streak[i]
                guess = await loop.run_in_executor(None, sys.stdin.readline)
                guess = guess.strip()

                if guess.upper() == "Q":
                    raise KeyboardInterrupt()

                if guess == expected:
                    print(f"  >> Correct! ({MOVE_NAMES[expected]})")
                else:
                    print(f"\n  >> WRONG! You chose {MOVE_NAMES.get(guess, '?')}, should be {MOVE_NAMES[expected]}")
                    print(f"\n  GAME OVER! Final score: {score}")
                    robot.send(sit_down())
                    await asyncio.sleep(2)
                    await robot.disconnect()
                    print("  Thanks for playing!\n")
                    return

            score += 1
            print(f"\n  >>> Round {len(streak)} complete! Score: {score}")
            print("  >>> Get ready for the next round...")
            await asyncio.sleep(2)

    except (EOFError, KeyboardInterrupt):
        print(f"\n\n  GAME OVER! Final score: {score}")

    # Clean shutdown
    robot.send(sit_down())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  Thanks for playing!\n")


if __name__ == "__main__":
    asyncio.run(main())
