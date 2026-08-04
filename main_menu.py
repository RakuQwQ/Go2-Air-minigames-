"""
Go2 Air Minigames - Main Menu Launcher
========================================
Choose from a collection of interactive mini-games for your Unitree Go2 Air robot.

Usage:
    python main_menu.py

Prerequisites:
    - Go2 Air robot powered on, connected via STA-L WiFi
    - UNITREE_AES_128_KEY environment variable set
    - Required libraries installed (see requirements.txt)
"""

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
from datetime import datetime


# Games registry
GAMES = [
    {
        "id": "1",
        "name": "Tug Commander",
        "module": "games.tug_commander",
        "desc": "Two-player hand gesture accuracy duel!",
    },
    {
        "id": "2",
        "name": "Pandora Message",
        "module": "games.pandora_message",
        "desc": "Voice + Hand Gesture dual control.",
    },
    {
        "id": "3",
        "name": "The Interpreter",
        "module": "games.the_interpreter",
        "desc": "Motion gesture control (dynamic).",
    },
]


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the main menu header."""
    clear_screen()
    print()
    print("  " + "=" * 56)
    print("  ||          GO2 AIR MINIGAMES               ||")
    print("  ||     Fun activities for your Go2 robot    ||")
    print("  " + "=" * 56)
    print()


def print_menu():
    """Print the game selection menu."""
    print_header()
    print("  Select a game to play:\n")
    for game in GAMES:
        print(f"    [{game['id']}]  {game['name']}")
        print(f"         {game['desc']}")
        print()
    print(f"    [0]  Exit")
    print()


def module_installed(module_name: str) -> bool:
    """Check whether a top-level import name is already installed."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def run_pip(args, label: str) -> int:
    """Run a pip command with a friendly status banner. Returns the return code."""
    print(f"\n  Installing {label} ...")
    print("  This may take a while (downloading + building wheels).")
    print("  " + "-" * 56)
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + args
    result = subprocess.run(cmd)
    print("  " + "-" * 56)
    if result.returncode == 0:
        print(f"  OK: {label} installed / updated successfully.")
    else:
        print(f"  WARN: {label} install FAILED (pip returned {result.returncode}).")
        print("        You can install it manually; see requirements.txt.")
    return result.returncode


CORE_PIP_REQ = [
    "git+https://github.com/tfoldi/go2-webrtc.git#subdirectory=python",
    "python-dotenv>=1.0.0",
]

# Each group is: (import_name_to_check, [pip_package_names], friendly_label, hint)
OPTIONAL_GROUPS = [
    ("numpy",              ["numpy>=1.24.0"],             "NumPy (math / vision helpers)", ""),
    ("cv2",                ["opencv-python>=4.8.0"],      "OpenCV (webcam / vision)", "Required by gesture & vision games"),
    ("mediapipe",          ["mediapipe>=0.10.0"],         "MediaPipe (hand tracking)", "Required by gesture-based games"),
    ("pygame",             ["pygame>=2.5.0"],             "Pygame (game controller)", "Required by the Joystick Pilot game"),
    ("colorama",           ["colorama>=0.4.6"],           "Colorama (colored terminal)", "Optional visual polish"),
    ("speech_recognition", ["SpeechRecognition>=3.10.0"], "SpeechRecognition (voice control)", "Required by the Pandora Message game"),
    ("pyaudio",            ["PyAudio"],                   "PyAudio (microphone audio)", "Required by the Pandora Message game"),
]


def install_core():
    """Install core dependencies (go2-webrtc + python-dotenv). Returns True on success."""
    print()
    print("  +----------------------------------------------+")
    print("  |  Dependency auto-check / auto-installer      |")
    print("  +----------------------------------------------+")
    print("\n  Checking core dependencies...")

    missing = []
    try:
        import unitree_webrtc_connect  # noqa: F401
    except ImportError:
        missing.append("go2-webrtc library (unitree_webrtc_connect)")
    if not module_installed("dotenv"):
        missing.append("python-dotenv")

    if not missing:
        print("  OK: core dependencies already present.")
        return True

    print("\n  Missing core dependencies:\n")
    for m in missing:
        print(f"    - {m}")

    print("\n  These are REQUIRED for all games (they talk to the Go2 robot).")
    answer = input("  Auto-install them now? [Y/n]: ").strip().lower()
    if answer in ("n", "no"):
        print("\n  Core dependencies not installed. The menu cannot connect to your robot.")
        print("  You can install them later with:\n    pip install -r requirements.txt")
        return False

    if not shutil.which("git"):
        print("\n  WARN: git is required to install the go2-webrtc library but was not found.")
        print("  Please install git (https://git-scm.com/downloads) and try again,")
        print("  or install manually with:\n")
        print("    pip install git+https://github.com/tfoldi/go2-webrtc.git#subdirectory=python")
        return False

    run_pip(CORE_PIP_REQ, "core dependencies")
    return True


def install_optionals():
    """Detect missing optional dependencies and offer to install each group."""
    print("\n  Checking optional dependencies...")

    missing_groups = []
    for check_import, pip_packages, label, hint in OPTIONAL_GROUPS:
        if not module_installed(check_import):
            missing_groups.append((check_import, pip_packages, label, hint))

    if not missing_groups:
        print("  OK: all optional dependencies are already installed.")
        return

    print(f"\n  Found {len(missing_groups)} optional package group(s) not installed:\n")
    for check_import, pip_packages, label, hint in missing_groups:
        print(f"    - {label}")
        if hint:
            print(f"      {hint}")

    print()
    answer = input("  Install ALL of them now? [Y/n]: ").strip().lower()
    if answer in ("n", "no"):
        print("\n  Skipping optional installs. Some games may ask to install on launch.")
        return

    for check_import, pip_packages, label, hint in missing_groups:
        if module_installed(check_import):
            continue
        if run_pip(pip_packages, label) != 0:
            print(f"  WARN: Could not install {label}; the affected game may not work.")


def ensure_dependencies():
    """Auto-install core deps (required) and offer to install optional deps up front."""
    if not install_core():
        return False
    install_optionals()
    return True


def check_env():
    """Check if required environment variables are set."""
    aes_key = os.getenv("UNITREE_AES_128_KEY", "")
    if not aes_key:
        print("\n  WARNING: UNITREE_AES_128_KEY not set!")
        print("  Some games may ask for it. Set it with:")
        print()
        print("    $env:UNITREE_AES_128_KEY = \"your-key-here\"")
        print()
        print("  Or create a .env file with:")
        print("    UNITREE_AES_128_KEY=your-key-here")
        print()


def launch_game(module_name: str):
    """
    Import and run a game module.
    The game module should have a `main()` async function.
    """
    try:
        game_module = importlib.import_module(module_name)
        if hasattr(game_module, "main"):
            asyncio.run(game_module.main())
        else:
            print(f"\n  ERROR: {module_name} has no main() function!")
            input("\n  Press Enter to return to menu...")
    except ImportError as e:
        print(f"\n  ERROR: Could not load game: {e}")
        print("  Make sure all dependencies are installed.")
        input("\n  Press Enter to return to menu...")
    except KeyboardInterrupt:
        print("\n\n  Game interrupted. Returning to menu...")
    except Exception as e:
        print(f"\n  ERROR in game: {type(e).__name__}: {e}")
        input("\n  Press Enter to return to menu...")


def main():
    """Main menu loop."""
    if not ensure_dependencies():
        print("\n  Dependency check incomplete; cannot continue.")
        input("\n  Press Enter to exit...")
        return

    check_env()

    while True:
        print_menu()

        try:
            choice = input("  Enter your choice [0-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if choice == "0":
            print("\n  Thanks for playing with your Go2! Goodbye!\n")
            break

        selected = None
        for game in GAMES:
            if game["id"] == choice:
                selected = game
                break

        if selected:
            print(f"\n  Launching: {selected['name']}...")
            launch_game(selected["module"])
        else:
            print("\n  Invalid choice. Please select a number from the menu.")
            input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()

