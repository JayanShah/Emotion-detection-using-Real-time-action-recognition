import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import sys
from src import config
from src.pose_wrapper import start_openpose, op
from src.tracker import SimpleTracker
from src.data_utils import process_buffer_to_tensor
from src.model import FinalModel, get_adjacency_matrix

def main():
    # 1. Load Model
    print(f"Loading Model from {config.MODEL_WEIGHTS_PATH}...")
    try:
        A = get_adjacency_matrix().to(config.DEVICE)
        model = FinalModel(A).to(config.DEVICE)
        
        checkpoint = torch.load(config.MODEL_WEIGHTS_PATH, map_location=config.DEVICE)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint.to(config.DEVICE)
        
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 2. Start OpenPose
    try:
        opWrapper = start_openpose(config.OPENPOSE_PARAMS)
    except Exception as e:
        print("OpenPose failed to start. Check src/config.py paths.")
        sys.exit(1)

    # 3. Setup Tracking & Buffers
    tracker = SimpleTracker(max_dist=config.TRACKER_MAX_DIST)
    buffers = {} 
    predictions = {} 

    cap = cv2.VideoCapture(0)
    window_name = "ST-GCN Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("System Ready. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # OpenPose Inference
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        keypoints = datum.poseKeypoints 

        if keypoints is not None and len(keypoints) > 0:
            
            # --- STEP A: TRACKING ---
            id_map = tracker.update(keypoints)

            for op_idx, stable_id in id_map.items():
                if stable_id not in buffers:
                    buffers[stable_id] = deque(maxlen=config.BUFFER_SIZE)
                    predictions[stable_id] = "Analyzing..."

                # --- STEP B: BUFFERING ---
                buffers[stable_id].append(keypoints[op_idx].copy())

                # --- STEP C: PREDICTION ---
                if len(buffers[stable_id]) == config.BUFFER_SIZE: 
                    input_tensor = process_buffer_to_tensor(buffers[stable_id])
                    input_tensor = input_tensor.to(config.DEVICE)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        conf, pred_idx = torch.max(probs, 1)
                        label = config.EMOTION_LABELS[pred_idx.item()]
                        predictions[stable_id] = f"{label} ({conf.item():.2f})"
                    
        # --- DRAWING ---
        if keypoints is not None:
            for op_idx, stable_id in id_map.items():
                pts = keypoints[op_idx, :, :2]
                pts = pts[pts[:, 0] > 0]
                if len(pts) > 0:
                    x_min, y_min = np.min(pts, axis=0).astype(int)
                    x_max, y_max = np.max(pts, axis=0).astype(int)
                    
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    label = predictions.get(stable_id, "Wait...")
                    cv2.putText(frame, f"ID {stable_id}: {label}", (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()