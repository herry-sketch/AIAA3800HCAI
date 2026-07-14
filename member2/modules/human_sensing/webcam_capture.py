from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

from .contracts import FramePacket


@dataclass(slots=True)
class CameraConfig:
    source_type: str = "opencv"
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    use_depth: bool = True
    align_depth_to_color: bool = True
    realsense_serial: str | None = None


class BaseCameraSource:
    def start(self) -> None:
        raise NotImplementedError

    def read(self) -> FramePacket:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "BaseCameraSource":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class OpenCVCameraSource(BaseCameraSource):
    def __init__(self, config: CameraConfig):
        self.config = config
        self._cv2 = None
        self._capture = None
        self._frame_index = 0

    def start(self) -> None:
        self._cv2 = importlib.import_module("cv2")
        self._capture = self._cv2.VideoCapture(self.config.device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open OpenCV camera index {self.config.device_index}")
        self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(self._cv2.CAP_PROP_FPS, self.config.fps)

    def read(self) -> FramePacket:
        if self._capture is None:
            raise RuntimeError("Camera has not been started")
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Failed to read frame from OpenCV camera")
        packet = FramePacket(
            timestamp=time.time(),
            color_frame=frame,
            frame_index=self._frame_index,
            source="opencv",
        )
        self._frame_index += 1
        return packet

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class RealSenseCameraSource(BaseCameraSource):
    def __init__(self, config: CameraConfig):
        self.config = config
        self._rs = None
        self._pipeline = None
        self._align = None
        self._frame_index = 0

    def start(self) -> None:
        self._rs = importlib.import_module("pyrealsense2")
        self._pipeline = self._rs.pipeline()
        cfg = self._rs.config()
        if self.config.realsense_serial:
            cfg.enable_device(self.config.realsense_serial)
        cfg.enable_stream(self._rs.stream.color, self.config.width, self.config.height, self._rs.format.bgr8, self.config.fps)
        if self.config.use_depth:
            cfg.enable_stream(self._rs.stream.depth, self.config.width, self.config.height, self._rs.format.z16, self.config.fps)
        self._pipeline.start(cfg)
        if self.config.use_depth and self.config.align_depth_to_color:
            self._align = self._rs.align(self._rs.stream.color)

    def read(self) -> FramePacket:
        if self._pipeline is None or self._rs is None:
            raise RuntimeError("RealSense pipeline has not been started")
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("Failed to read color frame from RealSense")
        color_image = importlib.import_module("numpy").asanyarray(color.get_data())
        depth_image = None
        if self.config.use_depth:
            depth = frames.get_depth_frame()
            if depth:
                depth_image = importlib.import_module("numpy").asanyarray(depth.get_data())
        intrinsics = color.profile.as_video_stream_profile().intrinsics
        packet = FramePacket(
            timestamp=time.time(),
            color_frame=color_image,
            depth_frame=depth_image,
            frame_index=self._frame_index,
            source="realsense",
            intrinsics={
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
            },
        )
        self._frame_index += 1
        return packet

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


def create_camera_source(config: CameraConfig) -> BaseCameraSource:
    source_type = config.source_type.lower()
    if source_type == "opencv":
        return OpenCVCameraSource(config)
    if source_type == "realsense":
        return RealSenseCameraSource(config)
    raise ValueError(f"Unsupported camera source_type: {config.source_type}")
