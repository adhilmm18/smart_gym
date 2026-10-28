"""
live_app.py - the Smart Gym Trainer (real-time).

Webcam -> MediaPipe Pose -> joint angles -> form classifier (Random Forest,
or rule-based fallback if no model is trained yet) -> rep counter ->
on-screen feedback + JSON log of every rep.

Run:    python live_app.py                      (webcam, starts with bicep curl)
        python live_app.py --exercise squat
        python live_app.py --source my_video.mp4

Keys:   1 = squat   2 = push-up   3 = bicep curl
        R = reset counters   S = save log   Q / ESC = quit
"""
import argparse
import json
import os
import time
from collections import deque
from datetime import datetime

import cv2
import joblib
import pandas as pd

from helpers import (BASE_DIR, EXERCISES, PRIMARY, REP_THRESHOLDS,
                     PoseEstimator, compute_features, draw_skeleton, put_text,
                     rule_check)

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
NAMES = {"squat": "Squat", "pushup": "Push-up", "curl": "Bicep Curl"}
GREEN, RED, WHITE, YELLOW = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 220, 255)


def load_model(exercise):
    path = os.path.join(MODEL_DIR, f"{exercise}_model.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


class ExerciseSession:
    """Rep counting state machine + posture classification for one exercise."""

    def __init__(self, exercise, use_model=True, cooldown=0.4):
        self.exercise = exercise
        self.ext, self.flex = REP_THRESHOLDS[exercise]
        self.primary = PRIMARY[exercise]
        self.bundle = load_model(exercise) if use_model else None
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.state = None            # None -> "extended" -> "flexed" -> "extended"
        self.active = False          # inside a rep (collecting form verdicts)
        self.flags = []              # per-frame "incorrect" flags during current rep
        self.min_angle = 999.0
        self.last_rep_t = -10.0
        self.reps = self.correct = self.incorrect = 0
        self.angle_buf = deque(maxlen=5)    # smooth the counting angle
        self.vote_buf = deque(maxlen=10)    # smooth the form verdict
        self.message, self.message_until = "", 0.0
        self.log = []

    @property
    def mode(self):
        return "ML model" if self.bundle else "rule-based (train a model for ML)"

    # ---- per-frame posture verdict ---------------------------------------
    def classify(self, feats):
        hints = rule_check(self.exercise, feats)
        if self.bundle:
            row = pd.DataFrame([[feats[k] for k in self.bundle["features"]]],
                               columns=self.bundle["features"])
            bad = self.bundle["model"].predict(row)[0] == "incorrect"
            if bad and not hints:
                hints = ["Check your form"]
        else:
            bad = bool(hints)
        return bad, hints

    # ---- main update ------------------------------------------------------
    def update(self, feats, t):
        """feats: dict of joint angles (or None). Returns (form_ok, hints, angle)."""
        if feats is None:
            return None, ["Step back so your body is fully visible"], None

        bad, hints = self.classify(feats)
        self.vote_buf.append(1 if bad else 0)
        form_ok = sum(self.vote_buf) / len(self.vote_buf) <= 0.5

        self.angle_buf.append(feats[self.primary])
        ang = sum(self.angle_buf) / len(self.angle_buf)

        if self.state is None:
            if ang > self.ext:
                self.state = "extended"
        elif self.state == "extended":
            if ang < self.ext - 10:
                self.active, self.flags, self.min_angle = True, [], ang
            if self.active:
                self.flags.append(bad)
                self.min_angle = min(self.min_angle, ang)
            if ang < self.flex:
                self.state = "flexed"
        elif self.state == "flexed":
            self.flags.append(bad)
            self.min_angle = min(self.min_angle, ang)
            if ang > self.ext and (t - self.last_rep_t) > self.cooldown:
                self._finish_rep(t)
        return form_ok, (hints if not form_ok else []), ang

    def _finish_rep(self, t):
        frac_bad = sum(self.flags) / max(len(self.flags), 1)
        verdict = "incorrect" if frac_bad > 0.3 else "correct"
        self.reps += 1
        if verdict == "correct":
            self.correct += 1
        else:
            self.incorrect += 1
        self.log.append({"rep": self.reps, "exercise": self.exercise,
                         "verdict": verdict,
                         "incorrect_frame_fraction": round(frac_bad, 2),
                         "min_angle": round(self.min_angle, 1),
                         "time": datetime.now().isoformat(timespec="seconds")})
        self.message = f"Rep {self.reps}: {verdict.upper()}"
        self.message_until = t + 1.5
        self.state, self.active, self.flags = "extended", False, []
        self.last_rep_t = t


def save_log(sessions):
    entries = [e for s in sessions.values() for e in s.log]
    if not entries:
        return None
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"session_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    return path


def draw_ui(frame, s, form_ok, hints, ang, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, 190), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    put_text(frame, NAMES[s.exercise], (10, 30), 0.9, YELLOW)
    put_text(frame, f"Reps: {s.reps}", (10, 62), 0.8)
    put_text(frame, f"Correct: {s.correct}", (10, 92), 0.7, GREEN)
    put_text(frame, f"Incorrect: {s.incorrect}", (10, 120), 0.7, RED)
    if form_ok is None:
        put_text(frame, "Posture: ---", (10, 152), 0.8, YELLOW)
    else:
        put_text(frame, "Posture: GOOD" if form_ok else "Posture: BAD", (10, 152),
                 0.8, GREEN if form_ok else RED)
    if ang is not None:
        put_text(frame, f"Angle: {ang:5.1f}  FPS: {fps:4.1f}", (10, 180), 0.55)

    y = 215
    for hint in hints[:3]:
        put_text(frame, hint, (10, y), 0.8, RED)
        y += 32
    put_text(frame, "1 squat | 2 push-up | 3 curl | R reset | S save | Q quit",
             (10, h - 15), 0.55)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", choices=EXERCISES, default="curl")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--no-model", action="store_true",
                    help="ignore trained models, use rules only")
    a = ap.parse_args()

    is_cam = str(a.source).isdigit()
    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    est = PoseEstimator()
    sessions = {ex: ExerciseSession(ex, not a.no_model) for ex in EXERCISES}
    current = a.exercise
    print(f"MediaPipe backend: {est.backend}")
    for ex in EXERCISES:
        print(f"  {ex:7s}: {sessions[ex].mode}")

    t0, prev, fps = time.time(), time.time(), 0.0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_cam:
            frame = cv2.flip(frame, 1)  # mirror view (angles are unaffected)

        now = time.time()
        ts = int((now - t0) * 1000) if is_cam else int(cap.get(cv2.CAP_PROP_POS_MSEC))
        ts = max(ts, last_ts + 1)
        last_ts = ts

        s = sessions[current]
        lms = est.process(frame, ts)
        feats = compute_features(current, lms)
        form_ok, hints, ang = s.update(feats, ts / 1000.0)

        draw_skeleton(frame, lms, GREEN if form_ok in (True, None) else RED)
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-3))
        prev = now
        draw_ui(frame, s, form_ok, hints, ang, fps)
        if s.message and ts / 1000.0 < s.message_until:
            col = GREEN if "CORRECT" in s.message and "INCORRECT" not in s.message else RED
            put_text(frame, s.message, (frame.shape[1] // 2 - 140, 60), 1.1, col, 3)

        cv2.imshow("Smart Gym Trainer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            current = EXERCISES[key - ord("1")]
            sessions[current].reset()
        elif key == ord("r"):
            s.reset()
        elif key == ord("s"):
            print("log saved:", save_log(sessions))

    cap.release()
    cv2.destroyAllWindows()
    path = save_log(sessions)
    if path:
        print("Session log saved:", path)
    for ex, s in sessions.items():
        if s.reps:
            print(f"{NAMES[ex]}: {s.reps} reps ({s.correct} correct, {s.incorrect} incorrect)")


if __name__ == "__main__":
    main()
