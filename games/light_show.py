"""
Game 8: Light Show
===================
Control your Go2 Air's LED headlights in fun patterns!

This game sends commands to control the robot's RGB LED lights
through the sport API.

Note: Light control commands may vary by firmware version.
This game demonstrates the concept and provides a framework.

Controls:
  1 = Red
  2 = Green
  3 = Blue
  4 = White
  5 = Rainbow (cycling)
  6 = Off
  P = Pattern: Blink
  L = Pattern: Pulse
  B = Pattern: Blink fast
  Q = Exit
"""

import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.robot_connector import RobotConnection
from utils.commands import stand_up, sit_down, generate_id
from unitree_webrtc_connect import SPORT_CMD


# Light colors (R, G, B)
COLORS = {
    "1": {"name": "Red",    "rgb": [255, 0, 0]},
    "2": {"name": "Green",  "rgb": [0, 255, 0]},
    "3": {"name": "Blue",   "rgb": [0, 0, 255]},
    "4": {"name": "White",  "rgb": [255, 255, 255]},
    "5": {"name": "Yellow", "rgb": [255, 255, 0]},
    "6": {"name": "Purple", "rgb": [128, 0, 128]},
    "7": {"name": "Cyan",   "rgb": [0, 255, 255]},
    "0": {"name": "Off",    "rgb": [0, 0, 0]},
}


def light_cmd(r: int, g: int, b: int, brightness: int = 100):
    """
    Generate a command to set the robot's headlight color.
    Note: This is an experimental command structure - adjust based on your firmware.
    """
    cmd = {
        "type": "msg",
        "topic": "rt/api/vui/request",
        "data": {
            "header": {"identity": {"id": generate_id(), "api_id": 0}},
            "parameter": json.dumps({
                "api_id": 0,
                "r": r,
                "g": g,
                "b": b,
                "brightness": brightness,
            }),
        },
    }
    return json.dumps(cmd)


async def main():
    print()
    print("  " + "=" * 56)
    print("  ||            LIGHT SHOW                  ||")
    print("  ||   LED light patterns for your Go2      ||")
    print("  " + "=" * 56)

    # Connect to robot
    robot = RobotConnection()
    if not await robot.connect():
        input("\n  Press Enter to exit...")
        return

    # Stand up
    robot.send(stand_up())

    print()
    print("  " + "=" * 50)
    print("  Colors:")
    for key, c in COLORS.items():
        print(f"    [{key}] {c['name']}")
    print()
    print("  Patterns:")
    print("    [P] Blink pattern   [L] Pulse pattern")
    print("    [B] Fast blink")
    print()
    print("  [Q] Exit")
    print("  " + "=" * 50)
    print()

    loop = asyncio.get_event_loop()

    try:
        while True:
            cmd = await loop.run_in_executor(None, sys.stdin.readline)
            cmd = cmd.strip().upper()

            if cmd == "Q":
                break

            elif cmd in COLORS:
                color = COLORS[cmd]
                print(f"  >> Setting color to {color['name']}")
                robot.send(light_cmd(*color["rgb"]))

            elif cmd == "P":
                print("  >> Blink pattern")
                for _ in range(3):
                    robot.send(light_cmd(255, 0, 0))
                    await asyncio.sleep(0.5)
                    robot.send(light_cmd(0, 0, 0))
                    await asyncio.sleep(0.5)

            elif cmd == "L":
                print("  >> Pulse pattern")
                for b in [50, 100, 150, 200, 150, 100, 50]:
                    robot.send(light_cmd(0, 0, 255, b))
                    await asyncio.sleep(0.3)

            elif cmd == "B":
                print("  >> Fast blink")
                for _ in range(5):
                    robot.send(light_cmd(255, 255, 255))
                    await asyncio.sleep(0.15)
                    robot.send(light_cmd(0, 0, 0))
                    await asyncio.sleep(0.15)

            else:
                print(f"  Unknown command: '{cmd}'")

    except (EOFError, KeyboardInterrupt):
        print("\n\n  >> Lights off!")

    # Clean shutdown
    robot.send(sit_down())
    await asyncio.sleep(2)
    await robot.disconnect()
    print("  >> Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
