## MiDaS Webcam Depth Map

This small project uses the Intel MiDaS model (via PyTorch Hub) and OpenCV to turn your webcam feed into a real-time depth (3D) map.

### Setup

- **Create / activate a virtual environment** (recommended).
- From the `CV Project` folder, install dependencies:

```bash
pip install -r requirements.txt
```

### Run

From the same folder (CLI version):

```bash
python midas_depth_webcam.py
```

Two windows will open: `Original Webcam` and `AI 3D Depth Map`.  
Press `q` to quit.

### Flask UI (recommended)

Start the web app:

```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

You will see a live stream in the browser: `Original | Depth (heatmap)`.

