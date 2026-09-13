#!/usr/bin/env python3
"""
Jetson_files/inference_benchmark.py

Standalone on-device inference latency benchmark for the acoustic (SmallCNN)
and optical (YOLO) models. Run directly on the Jetson against a recorded
test video / audio file - no camera, PTZ, or microphone hardware required.

Usage:
    python inference_benchmark.py --mode vision --video test_drone.mp4
    python inference_benchmark.py --mode acoustic --audio recordings/drone_sample.wav
    python inference_benchmark.py --mode both --video test_drone.mp4 --audio recordings/drone_sample.wav
"""
import os
import sys
import csv
import json
import time
import argparse
import threading
import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config
from uav_acoustic.model import SmallCNN


# ============================================================
#  Stats helpers
# ============================================================

def summarize(latencies_ms, unit_name="frame"):
    arr = np.array(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return None
    mean = float(arr.mean())
    return {
        "count": int(arr.size),
        "mean_ms": mean,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_per_sec": 1000.0 / mean if mean > 0 else 0.0,
        "unit": unit_name,
    }


def print_summary(title, stats):
    print(f"\n--- {title} ---")
    if stats is None:
        print("  [WARN] No samples measured (warmup >= total available frames/windows, or empty source).")
        return
    print(f"  Samples measured : {stats['count']}")
    print(f"  Average latency  : {stats['mean_ms']:.2f} ms")
    print(f"  Median latency   : {stats['median_ms']:.2f} ms")
    print(f"  P95 latency      : {stats['p95_ms']:.2f} ms")
    print(f"  Min / Max        : {stats['min_ms']:.2f} ms / {stats['max_ms']:.2f} ms")
    print(f"  Throughput       : {stats['throughput_per_sec']:.2f} {stats['unit']}/sec")


def parse_multi(value, cast=str):
    return [cast(v.strip()) for v in str(value).split(",") if v.strip() != ""]


# ============================================================
#  Vision (YOLO) benchmark
# ============================================================

def benchmark_vision_model(model_path, video_path, imgsz, conf, warmup,
                            raw_predict, realtime_pacing, stop_event=None):
    import cv2
    from ultralytics import YOLO

    print(f"\n[Vision] Loading model: {model_path} (imgsz={imgsz})")
    model = YOLO(model_path)
    if str(model_path).endswith(".pt"):
        # .to(device) only applies to native PyTorch weights - exported formats
        # (.engine/.onnx) are bound to a device/precision at export time.
        model.to(config.DEVICE)
    use_half = (config.DEVICE.type == "cuda")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Vision] Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / native_fps if native_fps > 0 else 0.0

    def run_once(frame):
        if raw_predict:
            # "Pure YOLO" number - bare forward pass, no ByteTrack overhead.
            return model.predict(frame, imgsz=imgsz, conf=conf,
                                  half=use_half, device=config.DEVICE, verbose=False)
        # Mirrors the exact production call in optical_processor.run_inference().
        return model.track(frame, imgsz=imgsz, stream=False, persist=True,
                            tracker="bytetrack.yaml", half=use_half, conf=conf,
                            device=config.DEVICE, verbose=False)

    # Warmup: first inference(s) pay one-time CUDA/TensorRT context and
    # conv+bn fusion costs that would otherwise skew the measured average.
    ret, frame = cap.read()
    warm_count = 0
    while ret and warm_count < warmup:
        run_once(frame)
        warm_count += 1
        ret, frame = cap.read()

    latencies = []
    next_due = time.perf_counter()
    while ret:
        if stop_event is not None and stop_event.is_set():
            break
        start = time.perf_counter()
        run_once(frame)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

        if realtime_pacing and frame_interval > 0:
            next_due += frame_interval
            sleep_for = next_due - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        ret, frame = cap.read()

    cap.release()
    return summarize(latencies, unit_name="frame")


def run_vision_benchmark(args, stop_event=None, results_out=None):
    model_paths = parse_multi(args.vision_model)
    imgsz_list = parse_multi(args.imgsz, cast=int)
    if len(imgsz_list) == 1:
        imgsz_list = imgsz_list * len(model_paths)
    if len(imgsz_list) != len(model_paths):
        raise ValueError("--imgsz must be a single value or match the number of --vision-model paths")

    all_stats = []
    for model_path, imgsz in zip(model_paths, imgsz_list):
        stats = benchmark_vision_model(
            model_path=model_path, video_path=args.video, imgsz=imgsz,
            conf=args.conf, warmup=args.warmup, raw_predict=args.raw_predict,
            realtime_pacing=args.realtime_pacing, stop_event=stop_event,
        )
        print_summary(f"VISION | {os.path.basename(model_path)} (imgsz={imgsz})", stats)
        entry = {"model": model_path, "imgsz": imgsz, "stats": stats}
        all_stats.append(entry)
        if results_out is not None:
            results_out.append(entry)
    return all_stats


# ============================================================
#  Acoustic (SmallCNN) benchmark
# ============================================================

def compute_logmel(y, sr, n_fft, hop_length, n_mels, fixed_length):
    """Exact same math as AcousticDetector.compute_live_logmel (acoustic_processor.py)."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                        hop_length=hop_length, n_mels=n_mels, power=2.0)
    spec_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    if spec_db.shape[1] < fixed_length:
        spec_db = np.pad(spec_db, ((0, 0), (0, fixed_length - spec_db.shape[1])))
    else:
        spec_db = spec_db[:, :fixed_length]
    return spec_db


def load_acoustic_model(checkpoint_path):
    import torch
    device = config.DEVICE
    model = SmallCNN(n_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"])


def benchmark_acoustic_model(checkpoint_path, audio_path, warmup, realtime_pacing, stop_event=None):
    import torch
    import librosa

    print(f"\n[Acoustic] Loading model: {checkpoint_path}")
    model, mean, std = load_acoustic_model(checkpoint_path)
    device = config.DEVICE

    sr = config.SAMPLE_RATE
    window_secs = getattr(config, "WINDOW_SECS", 1.0)
    step_secs = getattr(config, "STEP_SECS", 0.2)
    window_samples = int(sr * window_secs)
    step_samples = int(sr * step_secs)

    y, _ = librosa.load(audio_path, sr=sr)
    if len(y) < window_samples:
        y = np.pad(y, (0, window_samples - len(y)))

    starts = list(range(0, len(y) - window_samples + 1, step_samples))
    if not starts:
        starts = [0]

    def run_once(y_window):
        y_norm = y_window.copy()
        max_amp = np.max(np.abs(y_norm))
        if max_amp > 1e-8:
            y_norm /= max_amp
        mel = compute_logmel(y_norm, sr, config.MEL_N_FFT, config.MEL_HOP_LENGTH,
                              config.MEL_N_MELS, config.MEL_FIXED_LENGTH)
        x = torch.from_numpy(mel).float()
        x = (x - mean) / std
        x = x.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(x)

    for i in range(min(warmup, len(starts))):
        run_once(y[starts[i]:starts[i] + window_samples])

    latencies = []
    next_due = time.perf_counter()
    for start in starts[warmup:]:
        if stop_event is not None and stop_event.is_set():
            break
        t0 = time.perf_counter()
        run_once(y[start:start + window_samples])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        if realtime_pacing:
            # Matches acoustic_background_loop's real cadence: one CNN pass
            # every STEP_SECS, paced by the (here simulated) hardware read.
            next_due += step_secs
            sleep_for = next_due - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    return summarize(latencies, unit_name="window")


def run_acoustic_benchmark(args, stop_event=None, results_out=None):
    stats = benchmark_acoustic_model(
        checkpoint_path=args.acoustic_model, audio_path=args.audio,
        warmup=args.warmup, realtime_pacing=args.realtime_pacing, stop_event=stop_event,
    )
    print_summary(f"ACOUSTIC | {os.path.basename(args.acoustic_model)}", stats)
    entry = {"model": args.acoustic_model, "stats": stats}
    if results_out is not None:
        results_out.append(entry)
    return [entry]


# ============================================================
#  Output persistence
# ============================================================

def save_results(output_dir, mode, vision_results, acoustic_results, args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    payload = {
        "mode": mode,
        "timestamp": timestamp,
        "args": vars(args),
        "vision_results": vision_results,
        "acoustic_results": acoustic_results,
    }
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subsystem", "model", "imgsz", "samples", "mean_ms",
                          "median_ms", "p95_ms", "min_ms", "max_ms",
                          "throughput_per_sec", "unit"])
        for entry in vision_results:
            s = entry["stats"]
            if s:
                writer.writerow(["vision", entry["model"], entry.get("imgsz", ""), s["count"],
                                  f"{s['mean_ms']:.3f}", f"{s['median_ms']:.3f}", f"{s['p95_ms']:.3f}",
                                  f"{s['min_ms']:.3f}", f"{s['max_ms']:.3f}",
                                  f"{s['throughput_per_sec']:.3f}", s["unit"]])
        for entry in acoustic_results:
            s = entry["stats"]
            if s:
                writer.writerow(["acoustic", entry["model"], "", s["count"],
                                  f"{s['mean_ms']:.3f}", f"{s['median_ms']:.3f}", f"{s['p95_ms']:.3f}",
                                  f"{s['min_ms']:.3f}", f"{s['max_ms']:.3f}",
                                  f"{s['throughput_per_sec']:.3f}", s["unit"]])

    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


# ============================================================
#  CLI
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="On-device inference latency benchmark for the acoustic and/or vision models."
    )
    p.add_argument("--mode", required=True, choices=["vision", "acoustic", "both"])

    # Vision
    p.add_argument("--video", type=str, default=None,
                    help="Path to test .mp4 (required for --mode vision/both)")
    p.add_argument("--vision-model", type=str, default=config.YOLO_MODEL_PATH,
                    help="Comma-separated model path(s) (.engine/.pt/.onnx) - "
                         "pass several to compare resolutions/exports in one run")
    p.add_argument("--imgsz", type=str, default="640",
                    help="Single value applied to all models, or comma-separated to pair one-to-one")
    p.add_argument("--conf", type=float, default=config.YOLO_LOW_CONF_THRESHOLD)
    p.add_argument("--raw-predict", action="store_true",
                    help="Use plain model.predict() instead of the production track()+ByteTrack call")

    # Acoustic
    p.add_argument("--audio", type=str, default=None,
                    help="Path to test .wav (required for --mode acoustic/both)")
    p.add_argument("--acoustic-model", type=str,
                    default=os.path.join(SCRIPT_DIR, "uav_acoustic", "best_model.pt"))

    # Common
    p.add_argument("--warmup", type=int, default=20,
                    help="Frames/windows discarded before measuring (default: 20)")
    p.add_argument("--realtime-pacing", action="store_true",
                    help="Throttle vision to the video's native FPS and acoustic to STEP_SECS, "
                         "reproducing main_system.py's real duty cycle instead of max-throughput.")
    p.add_argument("--output-dir", type=str, default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode in ("vision", "both") and not args.video:
        parser.error("--video is required when --mode is 'vision' or 'both'")
    if args.mode in ("acoustic", "both") and not args.audio:
        parser.error("--audio is required when --mode is 'acoustic' or 'both'")

    print("=" * 60)
    print(" JETSON ON-DEVICE INFERENCE BENCHMARK")
    pacing_desc = ("real-time (matches main_system.py duty cycle)"
                   if args.realtime_pacing else "max throughput (worst-case ceiling)")
    print(f" Mode: {args.mode} | Device: {config.DEVICE} | Pacing: {pacing_desc}")
    print("=" * 60)

    vision_results, acoustic_results = [], []

    if args.mode == "vision":
        run_vision_benchmark(args, results_out=vision_results)

    elif args.mode == "acoustic":
        run_acoustic_benchmark(args, results_out=acoustic_results)

    elif args.mode == "both":
        # Mirrors main_system.py's real thread split: vision (optical_master_loop)
        # runs on the main thread, acoustic (acoustic_background_loop) runs on
        # one dedicated background thread. PTZ/FSM/watchdog threads are excluded
        # since they never touch the GPU or either model.
        print("\n[Both] Vision on the main thread, acoustic on one background thread - "
              "same split as main_system.py. NOTE: PTZ motor control and FSM logic are "
              "NOT simulated (no GPU/CNN involvement, and no physical hardware here).")
        stop_event = threading.Event()
        acoustic_thread = threading.Thread(
            target=run_acoustic_benchmark,
            args=(args,),
            kwargs={"stop_event": stop_event, "results_out": acoustic_results},
            name="AcousticBenchThread",
            daemon=True,
        )
        acoustic_thread.start()
        try:
            run_vision_benchmark(args, stop_event=stop_event, results_out=vision_results)
        finally:
            stop_event.set()
            acoustic_thread.join(timeout=30)

    save_results(args.output_dir, args.mode, vision_results, acoustic_results, args)

    print("\n[DONE] Benchmark complete.")


if __name__ == "__main__":
    main()
