# Smart Gym Trainer Using Pose Detection

Real-time exercise form checker and rep counter for **squats, push-ups and bicep curls**.
MediaPipe Pose (33 landmarks) -> joint angles -> Random Forest -> rep counter -> live feedback.
Runs fully offline on a laptop CPU.

## 1. Setup (Python 3.10 or 3.11)
```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```
First run downloads a ~6 MB pose model into `models/` if your MediaPipe version needs it.
If you are offline, download it from the URL printed in the error and save it at the path shown.

## 2. Try it right away (no training needed)
```
python live_app.py --exercise curl
```
Without trained models it uses rule-based form checks. Keys: 1 squat, 2 push-up, 3 curl, R reset, S save log, Q quit.

## 3. Build your dataset
Record videos (side view, whole body visible, 720p, fixed camera) and sort them like this:
```
videos/squat/correct/   videos/squat/incorrect/
videos/pushup/correct/  videos/pushup/incorrect/
videos/curl/correct/    videos/curl/incorrect/
```
Tips: use all 5 team members, different clothes/rooms/lighting, 10+ videos per class,
and name files by person (e.g. `adil_01.mp4`). Then:
```
python collect_data.py
```
Or record live: `python collect_data.py --webcam --exercise curl --label incorrect --person salvi`

## 4. Train
```
python train_model.py
```
Saves `models/<exercise>_model.joblib`, confusion-matrix PNGs and `models/report.json`
(accuracy / precision / recall / F1) for your report. The 80/20 split is done per video to avoid inflated accuracy.

## 5. Run the trained trainer
```
python live_app.py --exercise squat
```
Each completed rep is logged to `logs/session_*.json`.

## Tuning
In `helpers.py`: `REP_THRESHOLDS` (rep counting angles) and `rule_check()` (form-hint thresholds).
Adjust them on your own videos if reps are missed or hints trigger too often.

## Files
helpers.py (pose, angles, features) | collect_data.py | train_model.py | live_app.py
