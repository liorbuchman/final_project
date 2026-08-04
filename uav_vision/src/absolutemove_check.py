# check_getstatus_truth.py  (fixed)
import sys, os, time

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from camera_AC import setup_camera


def read_pan(ptz, token):
    status = ptz.GetStatus({'ProfileToken': token})
    return status.Position.PanTilt.x


def stop_camera(ptz, req, token):
    """Stop with two fallbacks - NEVER leave the camera spinning."""
    try:
        ptz.Stop({'ProfileToken': token, 'PanTilt': True, 'Zoom': False})
        return
    except Exception as e:
        print(f"  Stop() failed ({e}), falling back to zero-velocity ContinuousMove...")
    try:
        req.Velocity = {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}
        ptz.ContinuousMove(req)
    except Exception as e:
        print(f"  CRITICAL: zero-velocity fallback also failed: {e}")
        print("  STOP THE CAMERA MANUALLY via Web UI!")


def main():
    ptz, req, token, url = setup_camera()
    if ptz is None:
        return

    req.Velocity = {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}

    try:
        pos_before = read_pan(ptz, token)
        print(f"Position BEFORE move: {pos_before:.4f}")

        # Move AWAY from the -1.0 hard stop: positive pan
        print("Sending ContinuousMove (pan +0.3) for 3 seconds - WATCH the camera!")
        req.Velocity['PanTilt']['x'] = 0.3
        ptz.ContinuousMove(req)
        time.sleep(3.0)
    finally:
        # Stop runs even if anything above raises
        stop_camera(ptz, req, token)

    time.sleep(0.5)
    pos_after = read_pan(ptz, token)
    print(f"Position AFTER move:  {pos_after:.4f}")

    delta = abs(pos_after - pos_before)
    print(f"\nDelta: {delta:.4f}")
    if delta > 0.02:
        print("GetStatus reports REAL position -> feedback usable.")
        print("-> Dead-reckoning WITH poll correction is the path.")
    else:
        print("Camera moved but GetStatus did NOT change -> readback is FAKE.")
        print("-> Pure dead-reckoning + hard-stop homing required.")


if __name__ == "__main__":
    main()