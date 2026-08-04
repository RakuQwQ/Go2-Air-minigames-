"""
Shared robot connection logic for Go2 Air Minigames.

Provides a consistent way to connect to the Go2 robot across all games,
with proper error handling and cleanup.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Patch aiortc to skip audio/video transceivers (we don't need them)
from aiortc import RTCPeerConnection
_orig_add_transceiver = RTCPeerConnection.addTransceiver

def _patched_add_transceiver(self, kind, **kwargs):
    if kind in ('audio', 'video'):
        return None
    return _orig_add_transceiver(self, kind, **kwargs)

RTCPeerConnection.addTransceiver = _patched_add_transceiver

# Now safe to import the library
from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


logger = logging.getLogger(__name__)


class RobotConnection:
    """
    Manages the WebRTC connection to a Go2 robot.

    Usage:
        async with RobotConnection() as conn:
            # conn.dc is the data channel for sending commands
            conn.dc.send(some_command)
    """

    def __init__(self, robot_ip: str = None, aes_key: str = None):
        self.robot_ip = robot_ip or os.getenv("UNITREE_ROBOT_IP", "192.168.12.1")
        self.aes_key = aes_key or os.getenv("UNITREE_AES_128_KEY", "")
        self.conn = None
        self.dc = None

    async def connect(self, timeout: int = 20):
        """
        Connect to the robot and open the data channel.

        Args:
            timeout: Maximum seconds to wait for the data channel to open.

        Returns:
            True if connected successfully, False otherwise.
        """
        if not self.aes_key:
            print("\n  ERROR: AES-128 key not found!")
            print("  Set it as environment variable UNITREE_AES_128_KEY")
            print("  Or create a .env file (see .env.example)")
            print("  Get your key: unitree-fetch-aes-key --email your@email.com --region cn --device-type Go2\n")
            return False

        print(f"\n  >> Connecting to robot at {self.robot_ip}...")
        print(f"  >> AES key: {self.aes_key[:8]}...{self.aes_key[-4:]} ({len(self.aes_key)} chars)")

        try:
            self.conn = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=self.robot_ip,
                aes_128_key=self.aes_key,
            )
            await self.conn.connect()
        except Exception as e:
            print(f"\n  >> Connection failed: {type(e).__name__}: {e}")
            print("\n  Troubleshooting:")
            print("  1. Is the robot powered on?")
            print("  2. Are you on the same WiFi (192.168.12.x)?")
            print("  3. Is the AES key correct?")
            print("  4. Is another app already connected?")
            return False

        # Wait for data channel to open
        print("  >> Waiting for data channel...")
        for i in range(timeout * 2):
            await asyncio.sleep(0.5)
            if (self.conn.datachannel and
                self.conn.datachannel.channel and
                self.conn.datachannel.channel.readyState == "open"):
                self.dc = self.conn.datachannel.channel
                print("  >> Connected! Data channel is open.")
                return True

        print("\n  >> Data channel did not open within timeout.")
        await self.disconnect()
        return False

    def send(self, message: str) -> bool:
        """
        Send a command string over the data channel.

        Args:
            message: JSON command string to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.dc:
            print("  >> Error: No data channel. Not connected?")
            return False
        try:
            self.dc.send(message)
            return True
        except Exception as e:
            print(f"  >> Send error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the robot gracefully."""
        if self.conn:
            try:
                await self.conn.disconnect()
            except Exception:
                pass
            self.conn = None
            self.dc = None

    # Context manager support
    async def __aenter__(self):
        success = await self.connect()
        if not success:
            raise ConnectionError("Failed to connect to robot")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
