"""
train_model.py - train one Random Forest form-classifier per exercise.

Reads  data/<exercise>.csv   (made by collect_data.py)
Writes models/<exercise>_model.joblib
       models/<exercise>_confusion_matrix.png
       models/report.json      (accuracy, precision, recall, F1, confusion matrix)

Important: the 80/20 split is done BY VIDEO, not by random frame. Neighbouring
frames of the same video are almost identical, so a random frame split leaks
information and gives fake ~99% accuracy.

Usage:   python train_model.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from helpers import BASE_DIR, EXERCISES, FEATURES, LABELS

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def split_by_video(df, feats, seed=42):
    X, y, groups = df[feats], df["label"], df["video"]
    if groups.nunique() >= 4:
        for s in range(seed, seed + 100):  # find a split where test has both classes
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
            tr, te = next(gss.split(X, y, groups))
            if y.iloc[te].nunique() == 2 and y.iloc[tr].nunique() == 2:
                return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], "by video"
    print("  ! Fewer than 4 videos (or classes are unbalanced across videos): "
          "falling back to a random frame split. Accuracy will look too high - "
          "record more videos.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed,
                                          stratify=y)
    return Xtr, Xte, ytr, yte, "by frame (fallback)"


def save_confusion_png(cm, exercise, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2), LABELS)
    ax.set_yticks(range(2), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{exercise} - confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    report = {}
    for ex in EXERCISES:
        path = os.path.join(DATA_DIR, f"{ex}.csv")
        if not os.path.exists(path):
            print(f"[{ex}] no data/{ex}.csv - skipped")
            continue
        df = pd.read_csv(path).dropna()
        feats = FEATURES[ex]
        print(f"\n=== {ex.upper()} ===  samples: {len(df)}  videos: {df['video'].nunique()}")
        print(df["label"].value_counts().to_string())
        if df["label"].nunique() < 2:
            print("  need BOTH correct and incorrect samples - skipped")
            continue

        Xtr, Xte, ytr, yte, how = split_by_video(df, feats)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                     class_weight="balanced", random_state=42,
                                     n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        acc = accuracy_score(yte, pred)
        cm = confusion_matrix(yte, pred, labels=LABELS)
        rep = classification_report(yte, pred, labels=LABELS, output_dict=True,
                                    zero_division=0)
        print(f"split: {how}   train: {len(Xtr)}   test: {len(Xte)}")
        print(f"accuracy: {acc:.3f}")
        print(classification_report(yte, pred, labels=LABELS, zero_division=0))
        print("confusion matrix (rows=actual, cols=predicted; correct, incorrect):")
        print(cm)
        imp = dict(zip(feats, clf.feature_importances_.round(3)))
        print("feature importance:", imp)

        joblib.dump({"model": clf, "features": feats},
                    os.path.join(MODEL_DIR, f"{ex}_model.joblib"))
        save_confusion_png(cm, ex, os.path.join(MODEL_DIR, f"{ex}_confusion_matrix.png"))
        report[ex] = {"split": how, "train_samples": int(len(Xtr)),
                      "test_samples": int(len(Xte)), "accuracy": float(acc),
                      "confusion_matrix": cm.tolist(), "per_class": rep,
                      "feature_importance": {k: float(v) for k, v in imp.items()}}

    with open(os.path.join(MODEL_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nModels and report saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
