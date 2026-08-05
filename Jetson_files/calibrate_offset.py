#!/usr/bin/env python3
# calibrate_offset.py
#
# Guided pan+tilt calibration tool.
#
# Supersedes the old single-scalar CAMERA_MOUNT_OFFSET approach. That approach
# read the camera's *self-reported* pan angle via GetStatus to compute an
# offset against the DOA reading — but GetStatus on this camera has been
# observed to return a constant, fake value regardless of real position, so
# any offset computed that way was meaningless.
#
# This tool never reads camera position from the camera. Instead it builds a
# direct `DOA angle -> preset token` lookup table: for each point, the
# operator manually aims the camera (ContinuousMove, full visual control)
# while a real sound source plays, the tool samples the DSP's hardware DOA
# register directly over USB, and the resulting (doa, preset) pair is saved.
# At runtime, the pan-search logic can look up the nearest DOA bucket and
# GotoPreset straight to it — no offset, no sign convention, no conversion
# factor needed.
#
# Each point also optionally captures 2-3 tilt "rungs" (level/up/down) as
# separate presets at the same pan anchor, seeding an elevation search grid.
#
# The process is fully resumable: results are written to JSON after every
# single point, and any point can be revisited and redone without disturbing
# the others.

import os
import sys
import cv2
import time
import json
import struct
import statistics
import datetime
import usb.core

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from uav_vision.camera_AC import setup_camera
from uav_acoustic.respeaker_usb_led import ReSpeakerV31Leds

CALIBRATION_PATH = os.path.join(script_dir, "pan_tilt_calibration.json")

# --- Pan anchor grid ---
# Placeholder spacing. Tighten/loosen once camera_capability_test.py has
# established real pan speed / FOV numbers (open item from the hardware
# findings — anchor density depends on horizontal FOV, not yet measured).
PAN_RANGE_DEG = (-160.0, 160.0)
PAN_STEP_DEG = 20.0

TILT_RUNGS = ["level", "up", "down"]

SPEED_LEVELS = {ord('1'): 0.3, ord('2'): 0.6, ord('3'): 1.0}

LED_COLOR_IDLE = 0x001100      # soft green — positioning, no rush
LED_COLOR_READY = 0x000033     # dim blue — about to sample, get ready
LED_COLOR_SAMPLING = 0xFF0000  # bright red — make noise NOW
LED_COLOR_DONE = 0x111111      # brief white-ish flash — point saved

# ReSpeaker USB identifiers (same device used for DOA + LEDs elsewhere in the project)
USB_VENDOR_ID = 0x2886
USB_PRODUCT_ID = 0x0018


# --------------------------------------------------------------------------- #
# DOA sampling (raw USB, mirrors uav_acoustic/acoustic_processor.py's approach)
# --------------------------------------------------------------------------- #

def find_respeaker():
    dev = usb.core.find(idVendor=USB_VENDOR_ID, idProduct=USB_PRODUCT_ID)
    if dev is None:
        print("[WARN] ReSpeaker DSP not found on USB bus. DOA sampling will return 0.0.")
    return dev


def read_doa_once(dev):
    if dev is None:
        return 0.0
    try:
        res = dev.ctrl_transfer(0xC0, 0, 0xC0, 21, 8, 500)
        if len(res) >= 4:
            return float(struct.unpack(b'ii', res)[0])
    except Exception:
        pass
    return 0.0


def sample_doa_median(dev, duration_s=1.5, interval_s=0.05):
    samples = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        samples.append(read_doa_once(dev))
        time.sleep(interval_s)
    if not samples:
        return 0.0, []
    return statistics.median(samples), samples


# --------------------------------------------------------------------------- #
# Camera control (self-contained — does not depend on camera_capability_test.py)
# --------------------------------------------------------------------------- #

def ensure_velocity(ptz, request, token):
    """Same zeep None-Velocity fix used in camera_capability_test.py. See that
    file's docstring for the root cause (camera_AC.setup_camera() leaves
    request.Velocity unset)."""
    if request.Velocity is not None:
        return
    status = ptz.GetStatus({'ProfileToken': token})
    request.Velocity = status.Position
    request.Velocity.PanTilt.space = None
    if getattr(request.Velocity, 'Zoom', None) is not None:
        request.Velocity.Zoom.space = None


def send_velocity(ptz, request, pan=0.0, tilt=0.0):
    request.Velocity.PanTilt.x = round(float(pan), 2)
    request.Velocity.PanTilt.y = round(float(tilt), 2)
    try:
        ptz.ContinuousMove(request)
    except Exception as e:
        print(f"[PTZ ERROR] {e}")


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


def jog_until_confirm(cap, ptz, request, title, instruction_lines):
    """Free pan/tilt jog. Returns 'confirm' (c), 'abort' (m), or 'redo' (x)."""
    speed = 0.3
    while True:
        frame = grab_frame(cap)
        lines = [f"speed={speed} | w/a/s/d=jog 1/2/3=speed space=stop c=confirm x=redo-aim m=abort-point"] + instruction_lines
        if frame is not None:
            cv2.imshow("Calibration", render(frame, title, lines))
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
        elif key == 32:
            send_velocity(ptz, request, 0, 0)
        elif key == ord('c'):
            send_velocity(ptz, request, 0, 0)
            return 'confirm'
        elif key == ord('x'):
            send_velocity(ptz, request, 0, 0)
            return 'redo'
        elif key == ord('m'):
            send_velocity(ptz, request, 0, 0)
            return 'abort'


# --------------------------------------------------------------------------- #
# Calibration data persistence
# --------------------------------------------------------------------------- #

def pan_buckets():
    lo, hi = PAN_RANGE_DEG
    n = int(round((hi - lo) / PAN_STEP_DEG)) + 1
    return [round(lo + i * PAN_STEP_DEG, 1) for i in range(n)]


def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("camera_ip", config.CAMERA_IP)
    points = data.setdefault("points", {})
    for i, bucket_deg in enumerate(pan_buckets()):
        key = str(i)
        if key not in points:
            points[key] = {
                "target_bucket_deg": bucket_deg,
                "confirmed": False,
                "pan_preset_token": None,
                "doa_deg": None,
                "tilt_rungs": {},
            }
        else:
            points[key]["target_bucket_deg"] = bucket_deg
            points[key].setdefault("tilt_rungs", {})
    save_calibration(data)
    return data


def save_calibration(data):
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_preset(ptz, token, name, existing_token=None):
    try:
        req = {'ProfileToken': token, 'PresetName': name}
        if existing_token:
            req['PresetToken'] = existing_token
        resp = ptz.SetPreset(req)
        preset_token = resp if isinstance(resp, str) else getattr(resp, 'PresetToken', existing_token)
        return preset_token
    except Exception as e:
        print(f"[ERROR] SetPreset('{name}') failed: {e}")
        return existing_token


# --------------------------------------------------------------------------- #
# Per-point calibration flow
# --------------------------------------------------------------------------- #

def led_set(leds, color_hex):
    if leds is not None:
        try:
            leds.mono(color_hex)
        except Exception:
            pass


def calibrate_pan_point(data, idx, cap, ptz, request, token, usb_dev, leds, include_tilt):
    key = str(idx)
    point = data["points"][key]
    bucket_deg = point["target_bucket_deg"]
    n_points = len(data["points"])

    while True:
        led_set(leds, LED_COLOR_IDLE)
        title = f"Point {idx + 1}/{n_points} | target bucket ~ {bucket_deg}deg"
        outcome = jog_until_confirm(
            cap, ptz, request, title,
            ["Aim camera at a sound source near this bearing, keep noise playing, then press c."]
        )
        if outcome == 'abort':
            print(f"Point {idx} aborted, left unconfirmed.")
            return
        if outcome == 'redo':
            continue

        led_set(leds, LED_COLOR_READY)
        time.sleep(0.4)
        led_set(leds, LED_COLOR_SAMPLING)
        print("Sampling DOA now — keep the sound playing...")
        doa_median, samples = sample_doa_median(usb_dev, duration_s=1.5)
        led_set(leds, LED_COLOR_IDLE)

        frame = grab_frame(cap)
        if frame is not None:
            cv2.imshow("Calibration", render(frame, title, [f"Sampled DOA (median) = {doa_median:.1f} deg"]))
            cv2.waitKey(300)

        print(f"Sampled DOA median: {doa_median:.1f} deg (n={len(samples)} samples)")
        accept = input("Accept this pan point? [y]es / [r]edo aim / [a]bort point: ").strip().lower()
        if accept == 'y':
            preset_name = f"acoustic_pan_{idx}"
            preset_token = set_preset(ptz, token, preset_name, existing_token=point.get("pan_preset_token"))
            point["pan_preset_token"] = preset_token
            point["doa_deg"] = doa_median
            point["confirmed"] = True
            point["confirmed_at"] = datetime.datetime.now().isoformat()
            save_calibration(data)
            led_set(leds, LED_COLOR_DONE)
            time.sleep(0.3)
            led_set(leds, LED_COLOR_IDLE)
            print(f"Saved pan preset '{preset_name}' (token={preset_token}).")
            break
        elif accept == 'a':
            print(f"Point {idx} aborted, left unconfirmed.")
            return
        # 'r' or anything else -> loop back to re-aim

    if include_tilt:
        calibrate_tilt_rungs(point, idx, cap, ptz, request, token)
        save_calibration(data)


def calibrate_tilt_rungs(point, idx, cap, ptz, request, token):
    for rung in TILT_RUNGS:
        existing = point["tilt_rungs"].get(rung, {}).get("preset_token")
        while True:
            title = f"Point {idx + 1} | tilt rung '{rung}'"
            outcome = jog_until_confirm(
                cap, ptz, request, title,
                [f"Jog tilt (w/s) to the '{rung}' elevation for this column, then press c. (x=skip this rung)"]
            )
            if outcome == 'abort':
                print(f"Tilt rung '{rung}' for point {idx} skipped.")
                break
            if outcome == 'redo':
                continue

            frame = grab_frame(cap)
            if frame is not None:
                cv2.imshow("Calibration", render(frame, title, [f"Confirm rung '{rung}'?"]))
                cv2.waitKey(300)
            accept = input(f"Save tilt rung '{rung}'? [y]es / [r]edo / [a]bort rung: ").strip().lower()
            if accept == 'y':
                preset_name = f"acoustic_pan_{idx}_tilt_{rung}"
                preset_token = set_preset(ptz, token, preset_name, existing_token=existing)
                point["tilt_rungs"][rung] = {
                    "preset_token": preset_token,
                    "confirmed_at": datetime.datetime.now().isoformat(),
                }
                print(f"Saved tilt preset '{preset_name}' (token={preset_token}).")
                break
            elif accept == 'a':
                break
            # 'r' -> re-aim this rung


# --------------------------------------------------------------------------- #
# Top-level menu
# --------------------------------------------------------------------------- #

def next_pending_index(data):
    for i in range(len(data["points"])):
        if not data["points"][str(i)]["confirmed"]:
            return i
    return None


def print_status(data):
    total = len(data["points"])
    done = sum(1 for p in data["points"].values() if p["confirmed"])
    print(f"\nProgress: {done}/{total} pan points confirmed.")
    for i in range(total):
        p = data["points"][str(i)]
        mark = "x" if p["confirmed"] else " "
        rungs = ",".join(sorted(p["tilt_rungs"].keys())) or "-"
        print(f"  [{mark}] {i:2d}  bucket~{p['target_bucket_deg']:6.1f}deg  doa={p['doa_deg']}  tilt_rungs={rungs}")
    print()


def main():
    print("--- Guided Pan/Tilt Acoustic Calibration Tool ---")

    print("Connecting to ReSpeaker DSP...")
    usb_dev = find_respeaker()

    print("Connecting to LEDs...")
    try:
        leds = ReSpeakerV31Leds()
    except Exception as e:
        print(f"[WARN] LEDs unavailable: {e}")
        leds = None

    print("Connecting to ONVIF camera...")
    ptz, request, token, ptz_url = setup_camera()
    if ptz is None:
        print("Camera connection failed! Aborting.")
        return
    ensure_velocity(ptz, request, token)

    cap = open_video()
    if not cap.isOpened():
        print("[WARN] Could not open video stream — you will be calibrating blind. Aborting.")
        return

    data = load_calibration()

    try:
        while True:
            print_status(data)
            print("[Enter]=next pending point   g <idx>=goto/redo a specific point   q=save & quit")
            cmd = input("> ").strip().lower()

            if cmd == 'q':
                break
            elif cmd == '':
                idx = next_pending_index(data)
                if idx is None:
                    print("All pan points are confirmed. Use 'g <idx>' to redo any point.")
                    continue
                calibrate_pan_point(data, idx, cap, ptz, request, token, usb_dev, leds, include_tilt=True)
            elif cmd.startswith('g'):
                parts = cmd.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("Usage: g <index>")
                    continue
                idx = int(parts[1])
                if idx < 0 or idx >= len(data["points"]):
                    print("Index out of range.")
                    continue
                calibrate_pan_point(data, idx, cap, ptz, request, token, usb_dev, leds, include_tilt=True)
            else:
                print("Unknown command.")
    finally:
        try:
            send_velocity(ptz, request, 0, 0)
        except Exception:
            pass
        if leds is not None:
            try:
                leds.off()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        save_calibration(data)
        print(f"\nCalibration data saved to {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
