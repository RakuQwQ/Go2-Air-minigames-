"""
Shared command helpers for Go2 Air Minigames.

Provides utility functions to generate properly formatted JSON commands
for the Go2 robot's WebRTC sport API.
"""

import json
import datetime
import random
from unitree_webrtc_connect import SPORT_CMD


def generate_id():
    """Generate a unique message ID for robot commands."""
    return int(
        datetime.datetime.now().timestamp() * 1000 % 2147483648
    ) + random.randint(0, 999)


def sport_cmd(api_id: int, parameter=None):
    """
    Generate a JSON command for the robot's sport API.

    Args:
        api_id: The API command ID (use SPORT_CMD constants).
        parameter: Optional dict of parameters for the command.

    Returns:
        JSON string ready to send over the data channel.
    """
    cmd = {
        "type": "msg",
        "topic": "rt/api/sport/request",
        "data": {
            "header": {
                "identity": {
                    "id": generate_id(),
                    "api_id": api_id
                }
            },
            "parameter": json.dumps(parameter or api_id),
        },
    }
    return json.dumps(cmd)


def move_cmd(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    """
    Generate a movement command.

    Args:
        x: Forward/backward velocity (positive = forward).
        y: Left/right strafe velocity.
        z: Rotational velocity (positive = turn left).

    Returns:
        JSON string ready to send over the data channel.
    """
    return sport_cmd(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})


def euler_cmd(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
    """
    Generate a body tilt (Euler) command.

    Args:
        roll: Side-to-side tilt (positive = tilt right).
        pitch: Forward/backward tilt.
        yaw: Rotational offset.

    Returns:
        JSON string ready to send over the data channel.
    """
    return sport_cmd(SPORT_CMD["Euler"], {"roll": roll, "pitch": pitch, "yaw": yaw})


def pose_cmd(joint_params: dict):
    """
    Generate a Pose command (API 1028) for individual joint control.

    Args:
        joint_params: Dict of joint names/indices to target angles.
                      e.g. {"FR_thigh": 1.5, "FR_calf": -2.2}

    Returns:
        JSON string ready to send over the data channel.
    """
    return sport_cmd(SPORT_CMD["Pose"], joint_params)


def stand_up():
    """Command the robot to stand up."""
    return sport_cmd(SPORT_CMD["BalanceStand"])


def sit_down():
    """Command the robot to sit down."""
    return sport_cmd(SPORT_CMD["Sit"])


def stop_moving():
    """Command the robot to stop all movement."""
    return sport_cmd(SPORT_CMD["StopMove"])


def dance(style: int = 1):
    """
    Command the robot to dance.

    Args:
        style: 1 = Dance1, 2 = Dance2
    """
    if style == 2:
        return sport_cmd(SPORT_CMD["Dance2"])
    return sport_cmd(SPORT_CMD["Dance1"])


def say_hello():
    """Command the robot to do the hello gesture."""
    return sport_cmd(SPORT_CMD["Hello"])


def finger_heart():
    """Command the robot to make a finger heart gesture."""
    return sport_cmd(SPORT_CMD["FingerHeart"])


def stretch():
    """Command the robot to stretch."""
    return sport_cmd(SPORT_CMD["Stretch"])


def wiggle_hips():
    """Command the robot to wiggle its hips."""
    return sport_cmd(SPORT_CMD["WiggleHips"])


# Map of friendly names to command functions
COMMANDS = {
    "stand": stand_up,
    "sit": sit_down,
    "stop": stop_moving,
    "dance1": lambda: dance(1),
    "dance2": lambda: dance(2),
    "hello": say_hello,
    "heart": finger_heart,
    "stretch": stretch,
    "wiggle": wiggle_hips,
}
