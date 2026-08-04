# preset_direction_test.py
# Hypothesis: GotoPreset can only drive the camera in ONE direction
# (rightward). Recall works only when the anchor is to the RIGHT of
# the current position.

import sys, os, time

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from camera_AC import setup_camera


def move_for(ptz, req, x_speed, seconds):
    req.Velocity = {'PanTilt': {'x': x_speed, 'y': 0.0}, 'Zoom': {'x': 0.0}}
    try:
        ptz.ContinuousMove(req)
        time.sleep(seconds)
    finally:
        req.Velocity = {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}
        ptz.ContinuousMove(req)


def recall_and_ask(ptz, token, preset_token, label):
    ptz.GotoPreset({'ProfileToken': token, 'PresetToken': preset_token})
    print(f"  [{label}] GotoPreset sent. Watch closely and wait ~8s.")
    moved = input("  Did the camera MOVE at all during recall? (y/n): ")
    arrived = input("  Is it now on the reference object? (y/o/n): ")
    print(f"  Recorded: moved={moved}, arrived={arrived}\n")


def main():
    ptz, req, token, url = setup_camera()
    if ptz is None:
        return

    print("Aim mid-range at a distinctive object.")
    input("Enter to save anchor...")
    preset_token = ptz.SetPreset({'ProfileToken': token, 'PresetName': 'DIR_TEST'})
    print(f"Anchor saved (token: {preset_token}).\n")

    # Round 1: camera LEFT of anchor -> recall must move RIGHT (predicted: works)
    print("--- Round 1: displace LEFT, recall needs RIGHTWARD motion ---")
    input("Enter...")
    move_for(ptz, req, -0.3, 3.0)
    recall_and_ask(ptz, token, preset_token, "from-left")

    # Round 2: camera RIGHT of anchor -> recall must move LEFT (predicted: fails)
    print("--- Round 2: displace RIGHT, recall needs LEFTWARD motion ---")
    input("Enter...")
    move_for(ptz, req, +0.3, 3.0)
    recall_and_ask(ptz, token, preset_token, "from-right")

    # Round 3: THE WORKAROUND - from the right, first force a leftward
    # overshoot PAST the anchor, then recall from the left.
    print("--- Round 3: workaround - overshoot left past anchor, then recall ---")
    input("Enter (camera should currently be right of anchor)...")
    move_for(ptz, req, -0.3, 6.0)   # long enough to pass the anchor leftward
    recall_and_ask(ptz, token, preset_token, "overshoot-then-recall")

    print("If Round 1 works, Round 2 fails, Round 3 works ->")
    print("unidirectional recall CONFIRMED, and the workaround is viable.")


if __name__ == "__main__":
    main()