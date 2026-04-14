# Emotion Detection Using Real-Time Action Recognition

> ⚠️ **Experimental Project** — This is a proof-of-concept system built to explore skeleton-based emotion recognition. It is not production-ready.

---

## Introduction

This project is a **real-time, multi-person emotion recognition system** that works by analyzing **body pose coordinates** rather than facial expressions. Using **OpenPose** for skeleton extraction and a custom **ST-GCN (Spatial Temporal Graph Convolutional Network)** model, the system predicts one of 7 emotions from a live webcam feed by studying how a person moves and holds their body over time.

The 7 recognized emotions are: `Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`.

The pipeline works as follows:

```
Webcam Feed → OpenPose (Skeleton Extraction) → Multi-Person Tracking → Frame Buffer → ST-GCN Model → Emotion Label
```

The system supports up to **5 people simultaneously** and assigns each person a stable tracked ID across frames.

---

## Limitations

- Works best when the subject is **front-facing** with the **full body clearly visible** to the camera.
- Requires **whole-body pose coordinates** for best prediction accuracy — partial detections will reduce reliability.
- Works in **dim lighting conditions**, but the body must be distinguishable enough for OpenPose to detect pose keypoints correctly.
- Currently, only the **front-view model** (`best_front.pth`) is available. Left-view and right-view model weights (`best_left.pth`, `best_right.pth`) are placeholders — **contributions for training and integrating these models are welcome!**
- This is an **experimental system** built as a proof of concept and should not be used in production environments.

---

## Project Structure

```
Emotion-detection-using-Real-time-action-recognition/
│
├── models/
|   ├── best_front.pth          
|   ├── best_left.pth           
|   └── best_right.pth  
|
├── src/
│   ├── __init__.py
│   ├── config.py           # All paths and parameters
│   ├── data_utils.py       # Keypoint normalization & feature engineering
│   ├── model.py            # ST-GCN model definition
│   ├── pose_wrapper.py     # OpenPose initialization wrapper
│   └── tracker.py          # Simple centroid-based multi-person tracker
│        
├── main.py                 # Main entry point
├── requirements.txt
└── README.md
```

---

## Requirements

### System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (64-bit) |
| Python | 3.7.9 |
| CUDA | 12.1 |
| GPU | NVIDIA GPU (CUDA-capable) |
| OpenPose | GPU version |

> CUDA 12.1 must be installed on your system. You can download it from the [NVIDIA CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive).

### Python Dependencies

- `torch==1.13.1+cu117`
- `torchvision==0.14.1+cu117`
- `torchaudio==0.13.1+cu117`
- `opencv-python`
- `numpy`
- `pyopenpose` *(installed via OpenPose build)*

---

## Installation Guide

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/Emotion-detection-using-Real-time-action-recognition.git
cd Emotion-detection-using-Real-time-action-recognition
```

### Step 2 — Set Up Python Virtual Environment

Make sure you have **Python 3.7.9** installed. Then create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Step 3 — Install PyTorch (CUDA 11.7)

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Step 4 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Install OpenPose (GPU Version)

OpenPose must be installed separately. You have two options:

**Option A — Download from Google Drive (Recommended)**

Download the pre-built OpenPose GPU folder directly:
> 📁 **https://drive.google.com/drive/folders/1H5o4AzIuHLxD2z_JOjt-ePAEV4FaqMFI?usp=drive_link**

> ⚠️ **Note:** The Drive folder contains only the **pre-built OpenPose GPU build**. The model weight files (`best_front.pth`, `best_left.pth`, `best_right.pth`) are **not included** — refer to Step 6 for those.

Extract and place the folder at:
```
C:\openpose
```

**Option B — Build from Official Website**

Download the OpenPose source and pre-trained pose models from the [official OpenPose website](https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_installation_0_index.html) and build it for Windows using CMake and Visual Studio.

Place or build OpenPose so that the final folder is located at:
```
C:\openpose
```

> **If you place OpenPose at a different location**, update the path in `src/config.py`:
> ```python
> OPENPOSE_PATH = r"C:\openpose"   # ← Change this
> ```

### Step 6 — Verify Model Weights

Ensure the following model weight files are present in the `models`:

```
best_front.pth     ← front-view trained model
best_left.pth      ← Left-view trained model
best_right.pth     ← Right-view trained model
```

If `best_front.pth` is missing, the system will not run.

---

## Running the System

Activate your virtual environment and run:

```bash
.venv\Scripts\activate
python main.py
```

The system will:
1. Load the ST-GCN model
2. Start OpenPose
3. Open your webcam feed
4. Begin detecting, tracking, and predicting emotions in real time

Press **`q`** to quit.

---

## Configuration

All key parameters can be adjusted in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENPOSE_PATH` | `C:\openpose` | Path to your OpenPose installation |
| `MODEL_WEIGHTS_PATH` | `best_front.pth` | Path to model weights |
| `DEVICE` | auto | `cuda` if GPU available, else `cpu` |
| `BUFFER_SIZE` | `50` | Number of frames before a prediction is made |
| `TRACKER_MAX_DIST` | `80` | Max pixel distance for person re-identification |
| `number_people_max` | `5` | Maximum number of people tracked simultaneously |

---

## Contributing

Contributions are welcome! Areas where help is especially needed:

- **Training left-view and right-view models** (`best_left.pth`, `best_right.pth`) to enable multi-view emotion prediction.
- Improving tracker robustness for crowded scenes.
- Dataset expansion and augmentation strategies.

Please open an issue or submit a pull request.

---

## References

[1] M. Zhang, Yanan Zhou, Xinye Xu, Ziwei Ren, Yihan Zhang, Shenglan Liu & Wenbo Luo, "Multi-view emotional expressions dataset (MEED) using 2D pose estimation," *Nature*, 2023. [Online]. Available: https://www.nature.com/articles/s41597-023-02551-y

[2] S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for skeleton-based action recognition," in *Proc. AAAI Conf. Artif. Intell.*, New Orleans, LA, USA, 2018, pp. 7444–7452.

---

## License

This project is intended for research and educational purposes only.
