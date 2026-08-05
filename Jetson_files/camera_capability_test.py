#!/usr/bin/env python3
# camera_capability_test.py
#
# TEMPORARY diagnostic tool (Jetson_files/) — NOT part of the runtime system.
#
# Purpose: build a fresh, visually-verified ground-truth report of what this
# ONVIF PTZ camera actually does vs. what it declares. Nothing here trusts a
# single ONVIF response as truth — every test requires the operator to
# confirm what actually happened on the live video feed.
#
# Run from Jetson_files/:  python camera_capability_test.py

import os
import sys
import cv2
import time
import json
import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from uav_vision.camera_AC import setup_camera, set_light_raw

REPORT_PATH = os.path.join(script_dir, "camera_capability_report.json")

SPEED_LEVELS = {ord('1'): 0.3, ord('2'): 0.6, ord('3'): 1.0}


# --------------------------------------------------------------------------- #
# Report persistence — every logged result is flushed to disk immediately so
# a crash or Ctrl+C never loses prior findings. Re-running a test overwrites
# only that test's entry.
# --------------------------------------------------------------------------- #

def load_report():
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"camera_ip": config.CAMERA_IP, "tests": {}}


def save_report(report):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def log_result(report, test_name, result):
    result["timestamp"] = datetime.datetime.now().isoformat()
    report["tests"][test_name] = result
    save_report(report)
    print(f"[LOGGED] {test_name} -> {result}")


# --------------------------------------------------------------------------- #
# Hardware bring-up
# --------------------------------------------------------------------------- #

def ensure_velocity(ptz, request, token):
    """
    Known zeep gotcha (confirmed by comparing camera_AC.setup_camera() against
    uav_vision/test_optical_local.py): ptz.create_type('ContinuousMove') leaves
    request.Velocity as None. Any attempt to set request.Velocity.PanTilt.x then
    raises AttributeError. Patch it locally without touching camera_AC.py.
    """
    if request.Velocity is not None:
        return False
    status = ptz.GetStatus({'ProfileToken': token})
    request.Velocity = status.Position
    request.Velocity.PanTilt.space = None
    if getattr(request.Velocity, 'Zoom', None) is not None:
        request.Velocity.Zoom.space = None
    return True


def send_velocity(ptz, request, pan=0.0, tilt=0.0, zoom=0.0):
    """Direct ContinuousMove call. Deliberately does NOT swallow exceptions —
    this tool exists to surface hardware faults, not hide them."""
    request.Velocity.PanTilt.x = round(float(pan), 2)
    request.Velocity.PanTilt.y = round(float(tilt), 2)
    if getattr(request.Velocity, 'Zoom', None) is not None:
        request.Velocity.Zoom.x = round(float(zoom), 2)
    ptz.ContinuousMove(request)


def open_video():
    pipeline = config.get_gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[WARN] GStreamer/RTSP pipeline did not open. Falling back to plain RTSP URL.")
        cap = cv2.VideoCapture(config.RTSP_URL)
    return cap


def grab_frame(cap):
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    frame = cv2.flip(frame, -1)
    return cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))


def render(frame, title, lines, color=(0, 255, 255)):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    y = out.shape[0] - 10 - (len(lines) - 1) * 18
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18
    return out


def ask_yes_no(prompt):
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"


def ask_text(prompt):
    return input(f"{prompt}: ").strip()


# --------------------------------------------------------------------------- #
# Shared jog loop — used by several tests. Runs until the operator presses
# 'm' (return to menu). Keys: w/s = tilt, a/d = pan, 1/2/3 = speed, space = stop.
# --------------------------------------------------------------------------- #

def jog_loop(cap, ptz, request, title, extra_lines, extra_key_handler=None):
    speed = 0.3
    while True:
        frame = grab_frame(cap)
        lines = [f"speed={speed} | w/a/s/d=jog 1/2/3=speed space=stop m=menu"] + extra_lines
        if frame is not None:
            cv2.imshow("Camera Capability Test", render(frame, title, lines))
        key = cv2.waitKey(30) & 0xFF

        if key in SPEED_LEVELS:
            speed = SPEED_LEVELS[key]
        elif key == ord('w'):
            send_velocity(ptz, request, 0, speed)
        elif key == ord('s'):
            send_velocity(ptz, request, 0, -speed)
        elif key == ord('a'):
            send_velocity(ptz, request, -speed, 0)
        elif key == ord('d'):
            send_velocity(ptz, request, speed, 0)
        elif key == 32:  # space
            send_velocity(ptz, request, 0, 0)
        elif key == ord('m'):
            send_velocity(ptz, request, 0, 0)
            return None
        elif extra_key_handler is not None:
            result = extra_key_handler(key, speed)
            if result is not None:
                send_velocity(ptz, request, 0, 0)
                return result


# --------------------------------------------------------------------------- #
# Individual capability tests
# --------------------------------------------------------------------------- #

def test_velocity_bug(report, ptz, request, token):
    print("\n=== TEST: ContinuousMove Velocity field diagnostic ===")
    was_broken = ensure_velocity(ptz, request, token)
    log_result(report, "velocity_none_bug", {
        "was_none_before_patch": was_broken,
        "note": "If True, camera_AC.setup_camera() would have silently failed on every "
                "move_camera() call (AttributeError swallowed by its try/except)."
    })
    print(f"Velocity was None before patch: {was_broken} (now patched for this session)\n")


def test_manual_jog(report, cap, ptz, request):
    print("\n=== TEST: ContinuousMove manual jog ===")
    print("Jog with w/a/s/d at each speed, watch the video, then return to menu (m).")

    def handler(key, speed):
        return None

    jog_loop(cap, ptz, request, "TEST 2: Manual Jog", ["Confirm motion direction/speed matches keys pressed."], handler)

    pan_ok = ask_yes_no("Did pan (a/d) move the camera visibly and match the expected left/right direction")
    tilt_ok = ask_yes_no("Did tilt (w/s) move the camera visibly and match the expected up/down direction")
    notes = ask_text("Any notes (sign flipped, one axis dead, jitter, etc.) or leave blank")
    log_result(report, "continuous_move_jog", {"pan_ok": pan_ok, "tilt_ok": tilt_ok, "notes": notes})


def test_get_status_truth(report, cap, ptz, request, token):
    print("\n=== TEST: GetStatus truthfulness ===")
    print("Jog the camera around; the reported PanTilt.x/y is overlaid live.")
    print("Watch whether the numbers actually change as you move. Press m when done.")

    last_status = {"x": None, "y": None}

    while True:
        frame = grab_frame(cap)
        try:
            status = ptz.GetStatus({'ProfileToken': token})
            last_status["x"] = round(status.Position.PanTilt.x, 4)
            last_status["y"] = round(status.Position.PanTilt.y, 4)
        except Exception as e:
            last_status["x"] = f"ERR: {e}"
            last_status["y"] = ""

        lines = [
            "w/a/s/d=jog 1/2/3=speed space=stop m=menu",
            f"GetStatus PanTilt.x={last_status['x']}  y={last_status['y']}",
        ]
        if frame is not None:
            cv2.imshow("Camera Capability Test", render(frame, "TEST 3: GetStatus Truthfulness", lines))
        key = cv2.waitKey(30) & 0xFF

        if key in SPEED_LEVELS:
            speed = SPEED_LEVELS[key]
        if key == ord('w'):
            send_velocity(ptz, request, 0, 0.3)
        elif key == ord('s'):
            send_velocity(ptz, request, 0, -0.3)
        elif key == ord('a'):
            send_velocity(ptz, request, -0.3, 0)
        elif key == ord('d'):
            send_velocity(ptz, request, 0.3, 0)
        elif key == 32:
            send_velocity(ptz, request, 0, 0)
        elif key == ord('m'):
            send_velocity(ptz, request, 0, 0)
            break

    changes_with_motion = ask_yes_no("Did GetStatus.x/y actually change as you moved (not stuck constant)")
    matches_real_position = ask_yes_no("If it changed, did the reported direction/magnitude look physically correct")
    notes = ask_text("Notes (e.g. 'stuck at -1.0', 'changes but wrong scale') or leave blank")
    log_result(report, "get_status_truthfulness", {
        "changes_with_motion": changes_with_motion,
        "matches_real_position": matches_real_position,
        "notes": notes,
    })


def test_preset_directionality(report, cap, ptz, request, token, axis_label="pan"):
    print(f"\n=== TEST: SetPreset/GotoPreset directionality ({axis_label}) ===")
    print("Jog to a LEFT anchor point, press 'p' to save it as preset A.")
    print("Then jog to a RIGHT anchor point, press 'p' to save it as preset B.")

    presets = {}

    def make_handler():
        def handler(key, speed):
            if key == ord('p'):
                name = ask_text("Preset label (e.g. 'A_left' or 'B_right')")
                try:
                    resp = ptz.SetPreset({'ProfileToken': token, 'PresetName': name})
                    token_val = resp if isinstance(resp, str) else getattr(resp, 'PresetToken', resp)
                    presets[name] = token_val
                    print(f"  Saved preset '{name}' -> token {token_val}")
                except Exception as e:
                    print(f"  [ERROR] SetPreset failed: {e}")
            return None
        return handler

    print("Press 'p' to save each anchor, 'm' when both A and B are saved.")
    jog_loop(cap, ptz, request, f"TEST 4: Preset Directionality ({axis_label})",
             ["p=save preset here, save A(left) then B(right), then m"], make_handler())

    if len(presets) < 2:
        print("Need at least 2 presets to test directionality. Skipping.")
        log_result(report, f"preset_directionality_{axis_label}", {"skipped": True, "reason": "fewer than 2 presets saved"})
        return

    names = list(presets.keys())
    name_a, name_b = names[0], names[1]
    results = {}

    for start_name, target_name in [(name_a, name_b), (name_b, name_a)]:
        print(f"\nGo to '{start_name}' first (approach manually), then we'll GotoPreset('{target_name}').")
        jog_loop(cap, ptz, request, f"Position at '{start_name}', then press m", [f"Manually park at '{start_name}', then m"])
        input(f"At '{start_name}'. Press Enter to fire GotoPreset('{target_name}')...")
        try:
            ptz.GotoPreset({'ProfileToken': token, 'PresetToken': presets[target_name]})
        except Exception as e:
            print(f"[ERROR] GotoPreset failed: {e}")
        time.sleep(2.0)
        frame = grab_frame(cap)
        if frame is not None:
            cv2.imshow("Camera Capability Test", render(frame, "Observe result", ["Did it move to the target preset? Check console."]))
            cv2.waitKey(500)
        arrived = ask_yes_no(f"Did the camera visibly arrive at '{target_name}' (approaching from '{start_name}')")
        results[f"from_{start_name}_to_{target_name}"] = arrived

    print("\nNow testing double GotoPreset with no movement in between (same target twice).")
    input("Press Enter to fire GotoPreset twice in a row on the same target...")
    try:
        ptz.GotoPreset({'ProfileToken': token, 'PresetToken': presets[name_b]})
        time.sleep(2.0)
        ptz.GotoPreset({'ProfileToken': token, 'PresetToken': presets[name_b]})
        time.sleep(1.0)
    except Exception as e:
        print(f"[ERROR] GotoPreset failed: {e}")
    double_call_ok = ask_yes_no("Did the second call behave sanely (no drift/no fault), same position as first")
    results["double_call_same_target_ok"] = double_call_ok

    log_result(report, f"preset_directionality_{axis_label}", results)


def test_absolute_move(report, cap, ptz, request, token):
    print("\n=== TEST: AbsoluteMove ===")
    candidates = [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.3)]
    results = {}
    for x, y in candidates:
        print(f"\nSending AbsoluteMove to PanTilt=({x},{y})...")
        try:
            abs_req = ptz.create_type('AbsoluteMove')
            abs_req.ProfileToken = token
            status = ptz.GetStatus({'ProfileToken': token})
            abs_req.Position = status.Position
            abs_req.Position.PanTilt.x = x
            abs_req.Position.PanTilt.y = y
            abs_req.Position.PanTilt.space = None
            ptz.AbsoluteMove(abs_req)
        except Exception as e:
            print(f"[ERROR] AbsoluteMove failed: {e}")
            results[f"{x}_{y}"] = {"error": str(e)}
            continue
        time.sleep(2.0)
        frame = grab_frame(cap)
        if frame is not None:
            cv2.imshow("Camera Capability Test", render(frame, "TEST 5: AbsoluteMove", [f"Target ({x},{y}) — check console"]))
            cv2.waitKey(500)
        moved = ask_yes_no(f"Did the camera visibly move for target ({x},{y})")
        results[f"{x}_{y}"] = {"moved": moved}
    log_result(report, "absolute_move", results)


def test_home_position(report, cap, ptz, request, token):
    print("\n=== TEST: SetHomePosition / GotoHomePosition ===")
    set_first = ask_yes_no("Jog to a 'home' spot manually first, then answer y to call SetHomePosition there")
    if set_first:
        jog_loop(cap, ptz, request, "Park at desired HOME position, then m", ["Then we'll call SetHomePosition"])
        try:
            ptz.SetHomePosition({'ProfileToken': token})
            print("SetHomePosition called.")
        except Exception as e:
            print(f"[ERROR] SetHomePosition failed: {e}")

    print("Now jog AWAY from that spot, then we'll call GotoHomePosition.")
    jog_loop(cap, ptz, request, "Jog away from home, then m", ["Then we'll call GotoHomePosition"])
    input("Press Enter to fire GotoHomePosition...")
    try:
        ptz.GotoHomePosition({'ProfileToken': token})
    except Exception as e:
        print(f"[ERROR] GotoHomePosition failed: {e}")
    time.sleep(2.0)
    returned = ask_yes_no("Did the camera visibly return to the home spot")
    log_result(report, "home_position", {"set_home_first": set_first, "returned_visually": returned})


def test_zoom(report, cap, ptz, request):
    print("\n=== TEST: Zoom (continuous) ===")
    print("Press 'i' to zoom in, 'o' to zoom out, space to stop, m to finish.")

    while True:
        frame = grab_frame(cap)
        lines = ["i=zoom in  o=zoom out  space=stop  m=menu"]
        if frame is not None:
            cv2.imshow("Camera Capability Test", render(frame, "TEST 6: Zoom", lines))
        key = cv2.waitKey(30) & 0xFF
        if key == ord('i'):
            send_velocity(ptz, request, 0, 0, 0.4)
        elif key == ord('o'):
            send_velocity(ptz, request, 0, 0, -0.4)
        elif key == 32:
            send_velocity(ptz, request, 0, 0, 0)
        elif key == ord('m'):
            send_velocity(ptz, request, 0, 0, 0)
            break

    zoom_visible = ask_yes_no("Did the image visibly zoom in/out (optical or digital)")
    notes = ask_text("Notes (only digital? no zoom range? error thrown?) or leave blank")
    log_result(report, "zoom", {"zoom_visible": zoom_visible, "notes": notes})


def test_speed_timing(report, cap, ptz, request):
    print("\n=== TEST: Pan speed timing (hard-stop to hard-stop) ===")
    print("For each speed, jog LEFT to the mechanical hard stop first (space to stop there).")
    results = {}
    for speed in (0.3, 0.6, 1.0):
        print(f"\n--- Speed {speed} ---")
        jog_loop(cap, ptz, request, f"Get to LEFT hard stop (any speed), then m", ["Use w/a/s/d to reach the LEFT hard stop"])
        input(f"At LEFT hard stop. Press Enter to start timed sweep RIGHT at speed {speed}...")
        t0 = time.time()
        send_velocity(ptz, request, speed, 0)
        print("Watch the video. Press ENTER the instant it hits the RIGHT hard stop.")
        input()
        send_velocity(ptz, request, 0, 0)
        elapsed = time.time() - t0
        print(f"Elapsed: {elapsed:.2f}s")
        results[str(speed)] = elapsed
    log_result(report, "pan_speed_timing_full_sweep_seconds", results)
    print("Raw seconds only — convert to deg/sec once the physical full-swing range (degrees) is known.")


def test_light_ir(report, ptz_url, token):
    print("\n=== TEST: Light / IR aux commands ===")
    for cmd in ("LightOn", "LightOff", "IrOn", "IrOff"):
        input(f"Press Enter to send '{cmd}'...")
        set_light_raw(ptz_url, token, cmd)
        ask_yes_no(f"Did '{cmd}' visibly change anything (press y/n after observing)")
    log_result(report, "light_ir_aux", {"tested": True, "note": "see console for per-command y/n"})


# --------------------------------------------------------------------------- #
# Menu
# --------------------------------------------------------------------------- #

MENU = """
==================== CAMERA CAPABILITY TEST ====================
 0. Velocity-field diagnostic (run this first)
 2. Manual jog (ContinuousMove) — confirm axes/directions
 3. GetStatus truthfulness (live overlay while jogging)
 4. SetPreset / GotoPreset directionality (pan)
 5. AbsoluteMove
 6. SetHomePosition / GotoHomePosition
 7. Zoom (continuous)
 8. SetPreset / GotoPreset directionality (tilt)
 9. Pan speed timing (hard-stop to hard-stop, per speed)
 L. Light / IR aux command test
 Q. Save report and quit
==================================================================
Report file: {report_path}
"""


def main():
    print("Connecting to camera...")
    ptz, request, token, ptz_url = setup_camera()
    if ptz is None:
        print("Could not connect to camera. Aborting.")
        return

    report = load_report()

    cap = open_video()
    if not cap.isOpened():
        print("[WARN] Could not open video stream — visual confirmation will not be possible.")

    try:
        while True:
            print(MENU.format(report_path=REPORT_PATH))
            choice = input("Select test: ").strip().upper()

            if choice == '0':
                test_velocity_bug(report, ptz, request, token)
            elif choice == '2':
                test_manual_jog(report, cap, ptz, request)
            elif choice == '3':
                test_get_status_truth(report, cap, ptz, request, token)
            elif choice == '4':
                test_preset_directionality(report, cap, ptz, request, token, axis_label="pan")
            elif choice == '5':
                test_absolute_move(report, cap, ptz, request, token)
            elif choice == '6':
                test_home_position(report, cap, ptz, request, token)
            elif choice == '7':
                test_zoom(report, cap, ptz, request)
            elif choice == '8':
                test_preset_directionality(report, cap, ptz, request, token, axis_label="tilt")
            elif choice == '9':
                test_speed_timing(report, cap, ptz, request)
            elif choice == 'L':
                test_light_ir(report, ptz_url, token)
            elif choice == 'Q':
                break
            else:
                print("Unknown option.")
    finally:
        try:
            send_velocity(ptz, request, 0, 0, 0)
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        save_report(report)
        print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
