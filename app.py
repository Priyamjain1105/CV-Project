import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template, request


app = Flask(__name__)


class MiDaSWebcamProcessor:
    """
    Loads MiDaS once, continuously reads the webcam, runs inference,
    and stores the latest processed JPEG frame for fast streaming.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_type: str = "MiDaS_small",
        display_width: int = 640,
        colormap: int = cv2.COLORMAP_MAGMA,
    ) -> None:
        self.camera_index = camera_index
        self.model_type = model_type
        self.display_width = display_width
        self.colormap = colormap

        self.cap = cv2.VideoCapture(self.camera_index)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

        self.latest_jpeg: Optional[bytes] = None
        self.ready = False
        self.error: Optional[str] = None

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas = None
        self.transform = None

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _load_model(self) -> Tuple[object, object]:
        # Avoid TorchHub interactive "trust repo" prompt inside the web thread.
        midas = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        midas.to(self.device)
        midas.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if self.model_type == "MiDaS_small":
            transform = midas_transforms.small_transform
        else:
            transform = midas_transforms.dpt_transform

        return midas, transform

    def _resize_keep_aspect(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if w == 0 or h == 0:
            return frame_bgr
        scale = self.display_width / float(w)
        new_w = self.display_width
        new_h = max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)

    def _predict_depth_color(self, frame_bgr_resized: np.ndarray) -> np.ndarray:
        # Convert to RGB for MiDaS transforms
        frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr_resized.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.detach().cpu().numpy()

        # Normalize to 0..255 for visualization
        output_norm = cv2.normalize(
            output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return cv2.applyColorMap(output_norm, self.colormap)

    def _run(self) -> None:
        try:
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam (camera index may be wrong).")

            self.midas, self.transform = self._load_model()
            self.ready = True
        except Exception as e:
            self.error = str(e)
            self.ready = False
            return

        # Main processing loop
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_resized = self._resize_keep_aspect(frame)
            depth_color = self._predict_depth_color(frame_resized)

            # Side-by-side view: Original | Depth
            combined = cv2.hconcat([frame_resized, depth_color])

            ok, buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            jpeg_bytes = buf.tobytes()
            with self.cond:
                self.latest_jpeg = jpeg_bytes
                self.cond.notify_all()

    def stop(self) -> None:
        self._running = False
        try:
            self.cap.release()
        except Exception:
            pass


_processor: Optional[MiDaSWebcamProcessor] = None
_processor_lock = threading.Lock()
_current_camera_index: int = 0


def get_processor(camera_index: Optional[int] = None) -> MiDaSWebcamProcessor:
    """
    Get the global processor. If a new camera index is requested,
    the existing processor is stopped and recreated with that index.
    """
    global _processor, _current_camera_index
    with _processor_lock:
        if camera_index is not None and camera_index != _current_camera_index:
            # Switch camera: stop current processor and create a new one.
            if _processor is not None:
                _processor.stop()
                _processor = None
            _current_camera_index = camera_index

        if _processor is None:
            _processor = MiDaSWebcamProcessor(camera_index=_current_camera_index)
            _processor.start()

        return _processor


@app.get("/")
def index():
    p = get_processor()
    return render_template(
        "index.html",
        device=str(p.device),
        camera_index=p.camera_index,
        model_type=p.model_type,
    )


@app.get("/status")
def status():
    p = get_processor()
    return jsonify(
        {
            "ready": p.ready,
            "model_type": p.model_type,
            "camera_index": p.camera_index,
            "device": str(p.device),
            "error": p.error,
        }
    )


@app.get("/video_feed")
def video_feed():
    p = get_processor()

    def generate():
        boundary = b"--frame"
        while True:
            if p.error:
                # If there's an error, stop streaming.
                break

            if not p.ready:
                time.sleep(0.2)
                continue

            with p.cond:
                # Wait for an updated frame, but keep a timeout to re-check error/ready.
                p.cond.wait(timeout=1.0)
                frame = p.latest_jpeg

            if frame is None:
                continue

            yield (
                boundary
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + b"\r\n"
                + frame
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/set_camera")
def set_camera():
    """
    Change the active webcam (camera index). The stream will restart with the new camera.
    """
    camera_raw = request.form.get("camera", "0")
    try:
        camera_index = int(camera_raw)
    except ValueError:
        camera_index = 0

    p = get_processor(camera_index=camera_index)

    return jsonify(
        {
            "ok": True,
            "camera_index": p.camera_index,
            "ready": p.ready,
            "error": p.error,
        }
    )


if __name__ == "__main__":
    # Important: disable reloader to avoid starting the webcam/model twice.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

