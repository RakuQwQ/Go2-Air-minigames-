"""
Game 5: Joystick Pilot
=======================
Control your Go2 Air with a game controller!

Requires:
  - pygame (for joystick/gamepad support)

Install with:
  pip install pygame

Controls:
  Left Stick (X axis) = Turn left/right
  Left Stick (Y axis) = Forward/backward
  Right Trigger        = Speed boost
  A Button             = Stand up
  B Button             = Sit down
  X Button             = Say hello
  Y Button             = Dance!
  LB (Left Bumper)     = Finger Heart
  RB (Right Bumper)    = Stop
  Start                = Exit
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

# Check for pygame
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# Button mappings (common Xbox/PS layout)
BUTTON_A = 0       # A (Xbox) / Cross (PS)
BUTTON_B = 1       # B (Xbox) / Circle (PS)
BUTTON_X = 2       # X (Xbox) / Square (PS)
BUTTON_Y = 3       # Y (Xbox) / Triangle (PS)
BUTTON_LB = 4      # Left Bumper
BUTTON_RB = 5      # Right Bumper
BUTTON_START = 7   # Start button

AXIS_LEFT_X = 0    # Left stick horizontal
AXIS_LEFT_Y = 1    # Left stick vertical
AXIS_RT = 5        # Right trigger


def print_controls():
    """Display joystick controls."""
    print()
    print("  " + "=" * 50)
    print("  JOYSTICK PILOT - Controls")
    print("  " + "=" * 50)
    print()
    print("  Left Stick = Move / Turn")
    print("  Right Trigger = Speed boost")
    print("  A = Stand up      B = Sit down")
    print("  X = Hello         Y = Dance")
    print("  LB = Heart        RB = Stop")
    print("  Start = Exit")
    print()


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||          JOYSTICK PILOT                ||")
    print("  ||   Control Go2 with a game controller!  ||")
    print("  " + "=" * 56)

    if not HAS_PYGAME:
        print()
        print("  This game requires pygame:")
        print("    pip install pygame")
        print()
        input("  Press Enter to return to menu...")
        return

    # Initialize pygame and joystick
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("\n  ERROR: No joystick/gamepad detected!")
        print("  Please connect a game controller and try again.")
        input("\n  Press Enter to return to menu...")
        pygame.quit()
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"\n  Detected controller: {joystick.get_name()}")
    print(f"  Axes: {joystick.get_numaxes()}, Buttons: {joystick.get_numbuttons()}")

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        pygame.quit()
        input("\n  Press Enter to exit...")
        return

    print_controls()
    print("  >>> Use your controller! Press Start to exit.\n")

    # Control state
    running = True
    clock = pygame.time.Clock()

    try:
        while running:
            pygame.event.pump()

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Check buttons
            if joystick.get_button(BUTTON_START):
                print("  >> Exiting...")
                running = False

            if joystick.get_button(BUTTON_A):
                robot.send(stand_up())
                print("  >> Stand up!")

            if joystick.get_button(BUTTON_B):
                robot.send(sit_down())
                print("  >> Sit down")

            if joystick.get_button(BUTTON_X):
                robot.send(say_hello())
                print("  >> Hello!")

            if joystick.get_button(BUTTON_Y):
                robot.send(dance(1))
                print("  >> Dance!")

            if joystick.get_button(BUTTON_LB):
                robot.send(finger_heart())
                print("  >> Heart! <3")

            if joystick.get_button(BUTTON_RB):
                robot.send(stop_moving())
                print("  >> Stop")

            # Left stick movement
            axis_x = joystick.get_axis(AXIS_LEFT_X)
            axis_y = joystick.get_axis(AXIS_LEFT_Y)
            rt = joystick.get_axis(AXIS_RT)

            # Dead zone
            dead_zone = 0.15
            if abs(axis_x) < dead_zone:
                axis_x = 0
            if abs(axis_y) < dead_zone:
                axis_y = 0

            # Speed multiplier (right trigger for boost)
            speed = 0.3 + max(0, -rt) * 0.4  # RT range: -1 to 1, negative = pressed

            # Only send move if there's meaningful input
            if abs(axis_x) > 0 or abs(axis_y) > 0:
                forward = -axis_y * speed  # Invert Y axis
                turn = axis_x * speed * 1.5
                robot.send(move_cmd(x=round(forward, 2), z=round(turn, 2)))
            else:
                robot.send(stop_moving())

            clock.tick(30)  # 30 Hz control loop
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n  >> Interrupted!")
    finally:
        pygame.quit()

    # Clean shutdown
    print("  >> Sitting down...")
    robot.send(sit_down())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  >> Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
