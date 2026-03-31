## Real-Time Webcam Depth Map (MiDaS + OpenCV)

This project turns a normal webcam feed into a live depth-style view using Intel's MiDaS model.
In simple words: it estimates which parts of the scene are closer or farther, then shows that as a color heatmap.

It is a lightweight, beginner-friendly project for learning practical computer vision with deep learning.

## What this project does

- Captures live video from your webcam.
- Runs each frame through a pre-trained MiDaS depth model.
- Converts depth output into a colored heatmap for easy visual understanding.
- Shows both views together:
  - `Original Webcam`
  - `AI 3D Depth Map`

## Tech stack

- Python
- PyTorch + TorchVision
- OpenCV
- NumPy
- timm (model dependency)
- Flask (for browser-based UI)

## Project structure

- `midas_depth_webcam.py` -> CLI app (desktop windows via OpenCV)
- `app.py` -> Flask web app
- `requirements.txt` -> Python dependencies

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

2. Activate it:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run options

### Option 1: CLI version (quick start)

```bash
python midas_depth_webcam.py
```

You will get two OpenCV windows: original feed and depth heatmap.
Press `q` to quit.

### Option 2: Flask web UI

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Notes

- The model estimates **relative depth**, not exact distance in meters.
- First run may take longer because PyTorch downloads MiDaS weights from Torch Hub.
- If CUDA is available, PyTorch can use GPU for faster performance.

## Troubleshooting

- **Webcam not opening**
  - Close apps that may already be using the camera (Zoom, Meet, etc.).
  - Try changing camera index in code from `cv2.VideoCapture(0)` to `1`.

- **Slow performance**
  - Close other heavy apps.
  - Use a smaller webcam resolution if needed.
  - Make sure CUDA is installed correctly if you want GPU acceleration.

- **Torch Hub model download issues**
  - Check internet connection.
  - Re-run once; temporary network issues are common.

## Why this project is useful

This is a great starter project if you want hands-on experience with:

- real-time computer vision pipelines,
- deep learning model inference,
- camera stream processing,
- and converting raw model output into user-friendly visuals.

---

If you want, the next step can be adding FPS display, depth smoothing, screenshots/video export, and object detection + depth together.

