"""
collect_data.py - build the joint-angle dataset (CSV) from videos or the webcam.

MODE 1 - recorded videos (recommended). Put videos in this layout:
    videos/squat/correct/amal_01.mp4
    videos/squat/incorrect/amal_02.mp4
    videos/pushup/correct/...
    videos/curl/incorrect/...
  then run:
    python collect_data.py

MODE 2 - record live from the webcam:
    python collect_data.py --webcam --exercise curl --label correct --person salvi
  SPACE = start/stop recording, Q = quit

Output: data/<exercise>.csv  with columns
    video, exercise, <joint angles...>, label
"""
import argparse
import os
import time

import cv2
import pandas as pd

from helpers import (BASE_DIR, EXERCISES, FEATURES, LABELS, PoseEstimator,
                     compute_features, draw_skeleton, put_text)

DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


def csv_path(exercise):
    return os.path.join(DATA_DIR, f"{exercise}.csv")


def append_rows(exercise, rows):
    if not rows:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    path = csv_path(exercise)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def already_done(exercise):
    path = csv_path(exercise)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["video"])["video"].unique())


def process_video(path, exercise, label, stride):
    """Extract one feature row every `stride` frames from a video file."""
    est = PoseEstimator()  # fresh estimator per video (timestamps restart)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows, i, vid = [], 0, os.path.basename(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        lms = est.process(frame, int(i * 1000 / fps))  # run every frame (tracking)
        if i % stride:
            continue
        f = compute_features(exercise, lms)
        if f is None:
            continue
        rows.append({"video": vid, "exercise": exercise, **f, "label": label})
    cap.release()
    return rows


def run_folder_mode(root, stride):
    for ex in EXERCISES:
        done = already_done(ex)
        for label in LABELS:
            folder = os.path.join(root, ex, label)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(VIDEO_EXT):
                    continue
                vid_id = name
                if vid_id in done:
                    print(f"skip (already in CSV): {ex}/{label}/{name}")
                    continue
                rows = process_video(os.path.join(folder, name), ex, label, stride)
                for r in rows:
                    r["video"] = vid_id
                append_rows(ex, rows)
                print(f"{ex:7s} {label:9s} {name:30s} -> {len(rows)} samples")
    print("\nDone. CSV files are in:", DATA_DIR)


def run_webcam_mode(exercise, label, person, source, stride):
    est = PoseEstimator()
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    recording, rows, seg, n, t0 = False, [], 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        lms = est.process(frame, int((time.time() - t0) * 1000))
        draw_skeleton(frame, lms)
        f = compute_features(exercise, lms)
        if recording:
            n += 1
            if f is not None and n % stride == 0:
                rows.append({"video": vid, "exercise": exercise, **f, "label": label})
        status = f"REC {len(rows)} samples" if recording else "paused"
        put_text(frame, f"{exercise} / {label}  [{status}]", (10, 30),
                 color=(0, 0, 255) if recording else (255, 255, 255))
        put_text(frame, "SPACE: start/stop   Q: quit", (10, 60), 0.6)
        cv2.imshow("Collect data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording:
                append_rows(exercise, rows)
                print(f"saved {len(rows)} samples ({vid})")
                rows = []
            else:
                seg += 1
                vid = f"{person}_{exercise}_{label}_{int(time.time())}_{seg}"
                n = 0
            recording = not recording
        elif key in (ord("q"), 27):
            break
    if recording and rows:
        append_rows(exercise, rows)
        print(f"saved {len(rows)} samples ({vid})")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=os.path.join(BASE_DIR, "videos"),
                    help="root folder of recorded videos")
    ap.add_argument("--stride", type=int, default=2,
                    help="keep 1 of every N frames (default 2)")
    ap.add_argument("--webcam", action="store_true", help="record live")
    ap.add_argument("--exercise", choices=EXERCISES)
    ap.add_argument("--label", choices=LABELS)
    ap.add_argument("--person", default="user", help="name tag for webcam clips")
    ap.add_argument("--source", default="0", help="camera index (webcam mode)")
    a = ap.parse_args()

    if a.webcam:
        if not (a.exercise and a.label):
            ap.error("--webcam needs --exercise and --label")
        run_webcam_mode(a.exercise, a.label, a.person, a.source, a.stride)
    else:
        run_folder_mode(a.folder, a.stride)
