from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import timm
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_WEIGHT_PATH = MODELS_DIR / "mobilevit_drowsiness.pth"
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HF_REPO_ID = "mosesb/drowsiness-detection-mobileViT-v2"
HF_TIMM_MODEL_ID = f"hf_hub:{HF_REPO_ID}"
ARCHITECTURE = "mobilevitv2_200"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
LABELS = {0: "Drowsy", 1: "Non Drowsy"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_local_architecture() -> torch.nn.Module:
    return timm.create_model(ARCHITECTURE, pretrained=False, num_classes=len(LABELS))


def normalize_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint did not contain a state_dict-compatible mapping.")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue
        clean_key = key
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalized[clean_key] = value

    if not normalized:
        raise ValueError("State dict was empty after normalization.")
    return normalized


def load_model() -> tuple[torch.nn.Module, str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_WEIGHT_PATH.exists():
        model = build_local_architecture()
        state_dict = normalize_state_dict(torch.load(LOCAL_WEIGHT_PATH, map_location="cpu"))
        model.load_state_dict(state_dict)
        return model, "local"

    try:
        model = timm.create_model(HF_TIMM_MODEL_ID, pretrained=True)
    except Exception as exc:
        raise RuntimeError(
            "Local weights were not found at "
            f"{LOCAL_WEIGHT_PATH} and downloading from Hugging Face failed. "
            "Place a valid state dict at models/mobilevit_drowsiness.pth or restore network access."
        ) from exc

    torch.save(model.state_dict(), LOCAL_WEIGHT_PATH)
    return model, "huggingface"


def ensure_face_landmarker_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if FACE_LANDMARKER_MODEL_PATH.exists():
        return FACE_LANDMARKER_MODEL_PATH

    temp_path = FACE_LANDMARKER_MODEL_PATH.with_suffix(".tmp")
    try:
        print(f"[INFO] Downloading face landmarker model to {FACE_LANDMARKER_MODEL_PATH}")
        urlretrieve(FACE_LANDMARKER_MODEL_URL, temp_path)
        temp_path.replace(FACE_LANDMARKER_MODEL_PATH)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            "MediaPipe face landmarker model is missing and automatic download failed. "
            "Place face_landmarker.task in models/ and retry."
        ) from exc

    return FACE_LANDMARKER_MODEL_PATH


def create_face_detector() -> tuple[str, Any]:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
        return "solutions", detector

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions

    model_path = ensure_face_landmarker_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = FaceLandmarker.create_from_options(options)
    return "landmarker", detector


def close_face_detector(detector: Any) -> None:
    if hasattr(detector, "close"):
        detector.close()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_fp16_available(model: torch.nn.Module, device: torch.device) -> bool:
    if device.type != "cuda":
        return False

    dummy = torch.zeros((1, 3, 224, 224), device=device, dtype=torch.float32)
    try:
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(dummy)
        return True
    except Exception as exc:
        print(f"[WARN] FP16 warmup failed, falling back to FP32: {exc}")
        return False


def preprocess_roi(face_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    chw = np.ascontiguousarray(chw)
    return torch.from_numpy(chw).unsqueeze(0)


def forward_logits(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16 and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            return model(tensor)


def predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    use_fp16: bool,
) -> tuple[dict[str, Any], bool]:
    actual_fp16 = use_fp16
    try:
        logits = forward_logits(model, tensor, device, use_fp16)
    except Exception as exc:
        if use_fp16 and device.type == "cuda":
            print(f"[WARN] FP16 inference failed, retrying in FP32: {exc}")
            logits = forward_logits(model, tensor, device, False)
            actual_fp16 = False
        else:
            raise

    probabilities = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    result = {
        "label": LABELS[predicted_index],
        "probabilities": probabilities,
        "predicted_index": predicted_index,
    }
    return result, actual_fp16


def square_box_from_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
    scale: float = 1.25,
) -> tuple[int, int, int, int] | None:
    if w <= 0 or h <= 0:
        return None

    side = int(round(max(w, h) * scale))
    side = max(2, min(side, frame_width, frame_height))

    center_x = x + w // 2
    center_y = y + h // 2
    left = center_x - side // 2
    top = center_y - side // 2
    left = max(0, min(left, frame_width - side))
    top = max(0, min(top, frame_height - side))
    right = left + side
    bottom = top + side
    return left, top, right, bottom


def square_box_from_solution_detection(
    detection: Any,
    frame_width: int,
    frame_height: int,
    scale: float = 1.25,
) -> tuple[int, int, int, int] | None:
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * frame_width)
    y = int(bbox.ymin * frame_height)
    w = int(bbox.width * frame_width)
    h = int(bbox.height * frame_height)
    return square_box_from_rect(x, y, w, h, frame_width, frame_height, scale)


def square_box_from_landmarks(
    landmarks: list[Any],
    frame_width: int,
    frame_height: int,
    scale: float = 1.25,
) -> tuple[int, int, int, int] | None:
    if not landmarks:
        return None

    x_values = [float(np.clip(landmark.x, 0.0, 1.0)) * frame_width for landmark in landmarks]
    y_values = [float(np.clip(landmark.y, 0.0, 1.0)) * frame_height for landmark in landmarks]
    x_min = int(min(x_values))
    y_min = int(min(y_values))
    x_max = int(max(x_values))
    y_max = int(max(y_values))
    return square_box_from_rect(
        x_min,
        y_min,
        max(1, x_max - x_min),
        max(1, y_max - y_min),
        frame_width,
        frame_height,
        scale,
    )


def pick_best_solution_box(
    detections: Any,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    if not detections:
        return None

    best_detection = max(detections, key=lambda det: det.score[0] if det.score else 0.0)
    return square_box_from_solution_detection(best_detection, frame_width, frame_height)


def pick_best_landmark_box(
    faces_landmarks: list[list[Any]],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    if not faces_landmarks:
        return None

    best_box = None
    best_area = -1
    for landmarks in faces_landmarks:
        box = square_box_from_landmarks(landmarks, frame_width, frame_height)
        if box is None:
            continue
        left, top, right, bottom = box
        area = max(0, right - left) * max(0, bottom - top)
        if area > best_area:
            best_box = box
            best_area = area
    return best_box


def detect_square_box(
    detector_backend: str,
    detector: Any,
    frame: np.ndarray,
    frame_width: int,
    frame_height: int,
    timestamp_ms: int,
) -> tuple[int, int, int, int] | None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if detector_backend == "solutions":
        detections = detector.process(rgb_frame).detections
        return pick_best_solution_box(detections, frame_width, frame_height)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_frame),
    )
    result = detector.detect_for_video(mp_image, timestamp_ms)
    return pick_best_landmark_box(result.face_landmarks, frame_width, frame_height)


class FpsTracker:
    def __init__(self) -> None:
        self.last_time = time.perf_counter()
        self.smoothed_fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        delta = max(now - self.last_time, 1e-6)
        instant = 1.0 / delta
        if self.smoothed_fps == 0.0:
            self.smoothed_fps = instant
        else:
            self.smoothed_fps = (0.9 * self.smoothed_fps) + (0.1 * instant)
        self.last_time = now
        return self.smoothed_fps


def draw_overlay(
    frame: np.ndarray,
    result_text: str,
    drowsy_score: float | None,
    non_drowsy_score: float | None,
    fps: float,
) -> None:
    lines = [
        f"Result: {result_text}",
        f"Drowsy: {drowsy_score:.2f}" if drowsy_score is not None else "Drowsy: --",
        f"Non Drowsy: {non_drowsy_score:.2f}" if non_drowsy_score is not None else "Non Drowsy: --",
        f"FPS: {fps:.1f}",
    ]
    y = 32
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30


def main() -> None:
    model, _ = load_model()
    device = get_device()
    model = model.to(device)
    model.eval()
    use_fp16 = check_fp16_available(model, device)

    detector_backend, face_detector = create_face_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        close_face_detector(face_detector)
        raise RuntimeError("Unable to open webcam at index 0.")

    fps_tracker = FpsTracker()
    frame_index = 0
    last_prediction: dict[str, Any] | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            timestamp_ms = int(time.perf_counter() * 1000)
            box = detect_square_box(
                detector_backend,
                face_detector,
                frame,
                frame_width,
                frame_height,
                timestamp_ms,
            )

            result_text = "No Face"
            drowsy_score = None
            non_drowsy_score = None

            if box is not None:
                left, top, right, bottom = box
                cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 255), 2)
                face_roi = frame[top:bottom, left:right]

                if face_roi.size > 0 and frame_index % 2 == 0:
                    input_tensor = preprocess_roi(face_roi)
                    last_prediction, use_fp16 = predict(model, input_tensor, device, use_fp16)

                if last_prediction is not None:
                    result_text = last_prediction["label"]
                    drowsy_score = float(last_prediction["probabilities"][0])
                    non_drowsy_score = float(last_prediction["probabilities"][1])

            fps = fps_tracker.update()
            draw_overlay(display, result_text, drowsy_score, non_drowsy_score, fps)

            cv2.imshow("MobileViT Drowsiness", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            frame_index += 1
    finally:
        cap.release()
        close_face_detector(face_detector)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
