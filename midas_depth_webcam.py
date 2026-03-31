import cv2
import torch
import numpy as np


def main() -> None:
    # 1. Load the MiDaS model from TorchHub (Intel's official repo)
    # 'MiDaS_small' is fast and works great on standard laptops
    model_type = "MiDaS_small"
    # Avoid TorchHub interactive "trust repo" prompt.
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

    # 2. Use GPU if available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # 3. Load transformations to resize the image for the AI
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform

    # 4. Open your Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB and Transform for the model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        # Predict the depth
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Normalize output to 0-255 (for display)
        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(
            output,
            None,
            0,
            255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        # Apply a color map so "depth" looks like a heatmap (Magma/Plasma)
        output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

        # Show the result
        cv2.imshow("Original Webcam", frame)
        cv2.imshow("AI 3D Depth Map", output_color)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

