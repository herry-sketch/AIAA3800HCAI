from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import LABELS, check_fp16_available, get_device, load_model, predict, preprocess_roi  # noqa: E402


def main() -> None:
    print("== MobileViT Drowsiness Model Check ==")

    model, source = load_model()
    print("Model loaded: True")
    print(f"Model source: {source}")

    device = get_device()
    cuda_available = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"CUDA available: {cuda_available}")
    print(f"GPU name: {gpu_name}")

    model = model.to(device)
    model.eval()
    use_fp16 = check_fp16_available(model, device)
    print(f"FP16 enabled: {use_fp16}")

    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    input_tensor = preprocess_roi(dummy_image)
    output, actual_fp16 = predict(model, input_tensor, device, use_fp16)
    print("Dummy inference: success")
    print(f"Dummy inference precision: {'FP16' if actual_fp16 else 'FP32'}")
    print(f"Predicted class: {output['label']}")
    print(
        "Probabilities: "
        f"Drowsy={output['probabilities'][0]:.4f}, "
        f"Non Drowsy={output['probabilities'][1]:.4f}"
    )
    print(f"Class mapping: {LABELS}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
