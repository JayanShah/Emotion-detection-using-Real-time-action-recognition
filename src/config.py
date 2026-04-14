import os
import torch

# Paths
OPENPOSE_PATH = r"C:\openpose"
MODEL_WEIGHTS_PATH = os.path.join("models", "best_front.pth")

# Compute Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# OpenPose Configuration
OPENPOSE_PARAMS = {
    "model_folder": os.path.join(OPENPOSE_PATH, "models"),
    "net_resolution": "-1x160",
    "number_people_max": 5,
    "display": "0"
}

# Labels
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Tracking
TRACKER_MAX_DIST = 80
BUFFER_SIZE = 50  # Number of frames to collect before prediction