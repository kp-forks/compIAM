from compiam.rhythm.beat_tracking import TCNTracker
import os

audio_path = "../Devi Pavane.wav"

tracker = TCNTracker(model_path=os.path.join("..", "models", "tcn_tracker", "multitracker_42.pth"), post_processor="joint")
pred = tracker(audio_path)

print(pred)
