"""
helpers.py - shared code for the Smart Gym Trainer.

Used by collect_data.py, train_model.py and live_app.py so that the features
seen during training are computed EXACTLY the same way as in the live app.

Contains:
  * PoseEstimator  - wraps MediaPipe Pose (works with both the older
                     `mp.solutions.pose` API and the newer Tasks API)
  * angle helpers  - joint angles from landmark coordinates
  * compute_features() - joint-angle feature vector for each exercise
  * rule_check()   - simple rule-based form hints (also used as fallback
                     when no trained model exists yet)
"""
import os
import urllib.request

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
EXERCISES = ["squat", "pushup", "curl"]
LABELS = ["correct", "incorrect"]

# Feature columns (joint angles) for each exercise, in a fixed order.
FEATURES = {
    "squat":  ["knee_angle", "hip_angle", "torso_angle"],
    "pushup": ["elbow_angle", "body_angle", "knee_angle"],
    "curl":   ["elbow_angle", "shoulder_angle", "torso_angle"],
}

# The angle that drives the repetition counter.
PRIMARY = {"squat": "knee_angle", "pushup": "elbow_angle", "curl": "elbow_angle"}

# (extended, flexed) thresholds in degrees for the rep state machine.
# A rep = extended -> flexed -> extended.  Tune these on your own videos.
REP_THRESHOLDS = {"squat": (160, 100), "pushup": (160, 90), "curl": (150, 50)}

# MediaPipe landmark indices for the left / right side of the body.
JOINT_IDX = {
    "L": dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
    "R": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28),
}

# Joints that must be visible for each exercise.
NEEDED = {
    "squat":  ["shoulder", "hip", "knee", "ankle"],
    "pushup": ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"],
    "curl":   ["shoulder", "elbow", "wrist", "hip"],
}

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
            (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
POSE_TASK_FILE = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")


# --------------------------------------------------------------------------
# Pose estimation
# --------------------------------------------------------------------------
class PoseEstimator:
    """Returns a (33, 3) array [x_px, y_px, visibility] per frame, or None."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.backend = None
        self._last_ts = -1

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            # Older MediaPipe versions
            self.pose = mp.solutions.pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            self.backend = "solutions"
        else:
            # Newer MediaPipe versions (Tasks API) need a .task model file
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._ensure_task_file()
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_TASK_FILE),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.backend = "tasks"

    @staticmethod
    def _ensure_task_file():
        if os.path.exists(POSE_TASK_FILE):
            return
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading MediaPipe pose model (one time, ~6 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, POSE_TASK_FILE)
        except Exception as e:
            raise SystemExit(
                f"Could not download the pose model: {e}\n"
                f"Download it manually from:\n  {MODEL_URL}\n"
                f"and save it as:\n  {POSE_TASK_FILE}")

    def process(self, frame_bgr, ts_ms=0):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            res = self.pose.process(rgb)
            if not res.pose_landmarks:
                return None
            pts = res.pose_landmarks.landmark
        else:
            ts_ms = max(int(ts_ms), self._last_ts + 1)  # must increase
            self._last_ts = ts_ms
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(img, ts_ms)
            if not res.pose_landmarks:
                return None
            pts = res.pose_landmarks[0]
        # Use PIXEL coordinates so angles are not distorted by aspect ratio.
        return np.array([[p.x * w, p.y * h, p.visibility] for p in pts])


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def angle(a, b, c):
    """Angle in degrees at point b formed by points a-b-c."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def vertical_angle(top, bottom):
    """Angle between the vector bottom->top and the vertical (0 = upright)."""
    v = np.asarray(top, float) - np.asarray(bottom, float)
    cos = -v[1] / (np.linalg.norm(v) + 1e-9)  # image y axis points down
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pick_side(lms, exercise):
    """Use the body side (left/right) that the camera sees best."""
    score = {}
    for side in ("L", "R"):
        score[side] = sum(lms[JOINT_IDX[side][j], 2] for j in NEEDED[exercise])
    return "L" if score["L"] >= score["R"] else "R"


def compute_features(exercise, lms, min_vis=0.5):
    """Joint-angle features for one frame. Returns dict or None if unreliable."""
    if lms is None:
        return None
    side = pick_side(lms, exercise)
    idx = JOINT_IDX[side]
    if min(lms[idx[j], 2] for j in NEEDED[exercise]) < min_vis:
        return None
    P = {name: lms[i, :2] for name, i in idx.items()}

    f = {}
    if exercise == "squat":
        f["knee_angle"] = angle(P["hip"], P["knee"], P["ankle"])
        f["hip_angle"] = angle(P["shoulder"], P["hip"], P["knee"])
        f["torso_angle"] = vertical_angle(P["shoulder"], P["hip"])
    elif exercise == "pushup":
        f["elbow_angle"] = angle(P["shoulder"], P["elbow"], P["wrist"])
        f["body_angle"] = angle(P["shoulder"], P["hip"], P["ankle"])
        f["knee_angle"] = angle(P["hip"], P["knee"], P["ankle"])
    elif exercise == "curl":
        f["elbow_angle"] = angle(P["shoulder"], P["elbow"], P["wrist"])
        f["shoulder_angle"] = angle(P["hip"], P["shoulder"], P["elbow"])
        f["torso_angle"] = vertical_angle(P["shoulder"], P["hip"])
    else:
        raise ValueError(f"Unknown exercise: {exercise}")
    return f


# --------------------------------------------------------------------------
# Rule-based form hints (fallback + explanation text)
# --------------------------------------------------------------------------
def rule_check(exercise, f):
    """Return a list of human-readable hints; empty list = form looks fine.
    Thresholds are starting points - tune them on your own recordings."""
    hints = []
    if exercise == "squat":
        if f["torso_angle"] > 65:
            hints.append("Keep your chest up")
    elif exercise == "pushup":
        if f["body_angle"] < 160:
            hints.append("Keep your body in a straight line")
        if f["knee_angle"] < 150:
            hints.append("Keep your legs straight")
    elif exercise == "curl":
        if f["shoulder_angle"] > 35:
            hints.append("Keep your elbow close to your body")
        if f["torso_angle"] > 20:
            hints.append("Do not swing your back")
    return hints


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_skeleton(frame, lms, color=(0, 255, 0)):
    if lms is None:
        return
    for a, b in SKELETON:
        if lms[a, 2] > 0.4 and lms[b, 2] > 0.4:
            cv2.line(frame, tuple(lms[a, :2].astype(int)),
                     tuple(lms[b, :2].astype(int)), color, 3)
    for i in {i for pair in SKELETON for i in pair}:
        if lms[i, 2] > 0.4:
            cv2.circle(frame, tuple(lms[i, :2].astype(int)), 5, (0, 0, 255), -1)


def put_text(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)
