import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


def default_device() -> str:
    """Return a sensible default device string based on CUDA availability."""
    try:
        import torch  # type: ignore

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone detection with YOLOv11: images, videos, or webcam. Optional tracking and heatmap."
    )

    # Core IO
    parser.add_argument(
        "--source",
        type=str,
        default="test.jpeg",
        help="Path to image/video/dir/glob or webcam index (e.g., '0').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jawji.pt",
        help="Path to model weights (.pt). Defaults to 'jawji.pt' if present.",
    )

    # Inference params
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Device to use, e.g., 'cpu', '0', 'cuda:0'. Default auto-detected.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Filter by class IDs (e.g., --classes 0 1).",
    )
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16) if supported.")

    # Output and display
    parser.add_argument("--project", type=str, default="runs", help="Project directory for outputs.")
    parser.add_argument("--name", type=str, default=f"predict-{timestamp()}", help="Run name for output folder.")
    parser.add_argument("--show", action="store_true", help="Show annotated frames in a window.")
    parser.add_argument("--save_txt", action="store_true", help="Save results in label files.")

    # Tracking
    parser.add_argument("--track", action="store_true", help="Enable multi-object tracking on video/webcam.")
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Tracker config path (e.g., 'bytetrack.yaml'). Uses Ultralytics default if omitted.",
    )

    # Heatmap (custom pipeline)
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate and overlay a detection heatmap for video/webcam.",
    )
    parser.add_argument(
        "--heatmap-decay",
        type=float,
        default=0.97,
        help="Exponential decay factor per frame (0<d<=1). Lower means shorter memory.",
    )
    parser.add_argument(
        "--heatmap-blur",
        type=int,
        default=61,
        help="Gaussian kernel size for heatmap smoothing (odd integer).",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.5,
        help="Overlay transparency (0..1). Higher = stronger heatmap.",
    )

    args = parser.parse_args()
    return args


def is_webcam_source(src: str) -> bool:
    # Treat as webcam if it's a small integer string and not an existing path
    if Path(src).exists():
        return False
    try:
        _ = int(src)
        return True
    except Exception:
        return False


def run_predict(model: YOLO, args: argparse.Namespace) -> Path:
    """Run standard prediction on any source path (image/video/dir/glob/webcam)."""
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        half=args.half,
        classes=args.classes,
        verbose=False,
    )
    # Ultralytics saves outputs under project/name
    out_dir = Path(args.project) / args.name
    print(f"Prediction saved to: {out_dir.resolve()}")
    return out_dir


def run_track(model: YOLO, args: argparse.Namespace) -> Path:
    """Run tracking on a video/webcam source using Ultralytics tracker."""
    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        half=args.half,
        classes=args.classes,
        persist=True,
        tracker=args.tracker,
        verbose=False,
    )
    out_dir = Path(args.project) / args.name
    print(f"Tracking saved to: {out_dir.resolve()}")
    return out_dir


def _add_kernel(heatmap: np.ndarray, center: tuple[int, int], kernel: np.ndarray) -> None:
    """Safely add a 2D kernel centered at 'center' into 'heatmap'."""
    h, w = heatmap.shape
    kh, kw = kernel.shape
    cy, cx = center

    y1 = max(0, cy - kh // 2)
    y2 = min(h, cy + kh // 2 + 1)
    x1 = max(0, cx - kw // 2)
    x2 = min(w, cx + kw // 2 + 1)

    ky1 = max(0, kh // 2 - cy)
    ky2 = ky1 + (y2 - y1)
    kx1 = max(0, kw // 2 - cx)
    kx2 = kx1 + (x2 - x1)

    if y2 > y1 and x2 > x1:
        heatmap[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]


def gaussian_kernel(size: int = 61, sigma: float | None = None) -> np.ndarray:
    size = int(size) if int(size) % 2 == 1 else int(size) + 1  # ensure odd
    if sigma is None:
        sigma = 0.15 * size
    k1 = cv2.getGaussianKernel(size, sigma)
    kernel = k1 @ k1.T
    # Normalize to peak 1.0
    kernel /= kernel.max() if kernel.max() > 0 else 1.0
    return kernel.astype(np.float32)


def run_heatmap(model: YOLO, args: argparse.Namespace) -> Path:
    """Custom heatmap pipeline: run detection per frame, accumulate heatmap, save overlay video and heatmap image."""
    # Open source
    if is_webcam_source(args.source):
        cap_index = int(args.source)
        cap = cv2.VideoCapture(cap_index)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = out_dir / "heatmap_output.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Heatmap accumulation
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Determine kernel size relative to frame size for nicer visualization
    base = max(21, int(min(width, height) * 0.05))
    kernel_size = base if base % 2 == 1 else base + 1
    kernel = gaussian_kernel(size=max(kernel_size, args.heatmap_blur))

    decay = float(args.heatmap_decay)
    alpha = float(args.heatmap_alpha)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Decay accumulated heatmap
            if 0 < decay < 1:
                heatmap *= decay

            # Run detection on current frame (numpy array supported)
            results = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                classes=args.classes,
                verbose=False,
                stream=False,
            )

            if not results:
                overlay = frame
            else:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2) in xyxy:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        _add_kernel(heatmap, (cy, cx), kernel)

                # Create color heatmap overlay
                hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 1 - alpha, hm_color, alpha, 0)

            if args.show:
                cv2.imshow("Drone Detection Heatmap", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            writer.write(overlay)
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # Save the final heatmap image
    hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_img_path = out_dir / "heatmap.png"
    cv2.imwrite(str(heatmap_img_path), hm_color)

    print(f"Heatmap video saved to: {out_path.resolve()}")
    print(f"Heatmap image saved to: {heatmap_img_path.resolve()}")
    return out_dir


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        # Provide helpful message and fallback suggestions
        fallback = None
        for candidate in ("jawji.pt", "best.pt"):
            if Path(candidate).exists():
                fallback = candidate
                break
        msg = (
            f"Model weights not found: {args.model}. "
            + (f"Consider using '{fallback}'." if fallback else "Place your weights .pt in the project root.")
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    # Load model
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Failed to load model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Execution modes
    if args.heatmap:
        out_dir = run_heatmap(model, args)
    elif args.track:
        out_dir = run_track(model, args)
    else:
        out_dir = run_predict(model, args)

    print(f"Outputs available in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
