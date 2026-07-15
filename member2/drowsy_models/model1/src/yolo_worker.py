from __future__ import annotations

import multiprocessing as mp
import queue
import sys
from dataclasses import dataclass

import numpy as np

from .types import ClassificationObservation
from .yolo_classifier import YoloClassifier


@dataclass
class _YoloTask:
    frame_id: int
    timestamp: float
    face_roi: np.ndarray | None


def _worker_main(
    local_path: str,
    repo_id: str,
    filename: str,
    imgsz: int,
    device: int | str,
    half: bool,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:  # pragma: no cover - native runtime path
    try:
        classifier = YoloClassifier(
            local_path=local_path,
            repo_id=repo_id,
            filename=filename,
            imgsz=imgsz,
            device=device,
            half=half,
        )
    except Exception as exc:
        print(f"YOLO worker init failed: {exc}", file=sys.stderr, flush=True)
        return

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if task is None:
            break
        result = classifier.predict(task.face_roi, task.timestamp)
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
        result_queue.put(result)


class YoloWorker:
    def __init__(
        self,
        *,
        local_path: str,
        repo_id: str,
        filename: str,
        imgsz: int,
        device: int | str,
        half: bool,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._task_queue: mp.Queue = self._ctx.Queue(maxsize=1)
        self._result_queue: mp.Queue = self._ctx.Queue(maxsize=1)
        self._stop_event: mp.Event = self._ctx.Event()
        self._latest_result: ClassificationObservation | None = None
        self._process = self._ctx.Process(
            target=_worker_main,
            kwargs={
                "local_path": local_path,
                "repo_id": repo_id,
                "filename": filename,
                "imgsz": imgsz,
                "device": device,
                "half": half,
                "task_queue": self._task_queue,
                "result_queue": self._result_queue,
                "stop_event": self._stop_event,
            },
            name="yolo-worker",
            daemon=True,
        )
        self._process.start()

    def submit(self, frame_id: int, timestamp: float, face_roi: np.ndarray) -> None:
        task = _YoloTask(frame_id=frame_id, timestamp=timestamp, face_roi=np.ascontiguousarray(face_roi))
        try:
            self._task_queue.put_nowait(task)
        except queue.Full:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                pass
            self._task_queue.put_nowait(task)

    def get_latest_result(self) -> ClassificationObservation | None:
        try:
            while True:
                self._latest_result = self._result_queue.get_nowait()
        except queue.Empty:
            pass
        return self._latest_result

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                pass
            self._task_queue.put_nowait(None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

