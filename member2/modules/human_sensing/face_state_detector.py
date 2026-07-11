from __future__ import annotations

import time
from dataclasses import dataclass

from .contracts import FaceLandmarks, FaceStateSignals, HeadPose, HumanSensingHistory
from .utils import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, clamp, eye_aspect_ratio, mouth_aspect_ratio


@dataclass(slots=True)
class FaceStateConfig:
    yawn_mar_threshold: float = 0.33
    yawn_min_duration_sec: float = 0.6
    eyes_closed_ear_threshold: float = 0.19
    head_down_pitch_threshold: float = 18.0
    max_abs_yaw_for_screen: float = 30.0
    max_abs_pitch_for_screen: float = 25.0
    max_abs_roll_for_screen: float = 20.0


class FaceStateDetector:
    def __init__(self, config: FaceStateConfig | None = None):
        self.config = config or FaceStateConfig()

    def estimate_screen_facing(self, head_pose: HeadPose) -> float:
        yaw_score = clamp(1.0 - abs(head_pose.yaw) / self.config.max_abs_yaw_for_screen, 0.0, 1.0)
        pitch_score = clamp(1.0 - abs(head_pose.pitch) / self.config.max_abs_pitch_for_screen, 0.0, 1.0)
        roll_score = clamp(1.0 - abs(head_pose.roll) / self.config.max_abs_roll_for_screen, 0.0, 1.0)
        return round(0.5 * yaw_score + 0.4 * pitch_score + 0.1 * roll_score, 3)

    def detect_face_state_signals(
        self,
        face_landmarks: FaceLandmarks,
        history: HumanSensingHistory,
        head_pose: HeadPose | None = None,
        timestamp: float | None = None,
    ) -> FaceStateSignals:
        now = time.time() if timestamp is None else timestamp
        history.prune(now)

        if not face_landmarks.face_detected:
            history.current_yawn_started_at = None
            history.eye_closed_since = None
            return FaceStateSignals(
                timestamp=now,
                face_detected=False,
                screen_facing_score=0.0,
                yawn_detected=False,
                yawn_count_last_3min=len(history.yawn_timestamps),
                eyes_closed=False,
                eye_closure_duration_sec=0.0,
                head_down=False,
                mouth_aspect_ratio=0.0,
                eye_aspect_ratio=0.0,
                evidence=["face_not_detected"],
            )

        mar = mouth_aspect_ratio(face_landmarks.points)
        left_ear = eye_aspect_ratio(face_landmarks.points, LEFT_EYE_INDICES)
        right_ear = eye_aspect_ratio(face_landmarks.points, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        yawn_detected = False
        if mar >= self.config.yawn_mar_threshold:
            if history.current_yawn_started_at is None:
                history.current_yawn_started_at = now
            elif now - history.current_yawn_started_at >= self.config.yawn_min_duration_sec:
                if not history.yawn_timestamps or now - history.yawn_timestamps[-1] >= self.config.yawn_min_duration_sec:
                    history.yawn_timestamps.append(now)
                yawn_detected = True
        else:
            history.current_yawn_started_at = None

        eyes_closed = avg_ear <= self.config.eyes_closed_ear_threshold
        if eyes_closed:
            if history.eye_closed_since is None:
                history.eye_closed_since = now
            eye_closure_duration = now - history.eye_closed_since
        else:
            history.eye_closed_since = None
            eye_closure_duration = 0.0

        if head_pose is None:
            screen_facing_score = 0.0
            head_down = False
        else:
            screen_facing_score = self.estimate_screen_facing(head_pose)
            head_down = head_pose.pitch >= self.config.head_down_pitch_threshold

        evidence = [
            f"mouth_aspect_ratio={mar:.3f}",
            f"eye_aspect_ratio={avg_ear:.3f}",
            f"screen_facing_score={screen_facing_score:.3f}",
        ]
        if yawn_detected:
            evidence.append("yawn_detected")
        if eyes_closed:
            evidence.append(f"eye_closure_duration={eye_closure_duration:.2f}s")
        if head_down:
            evidence.append("head_down")

        return FaceStateSignals(
            timestamp=now,
            face_detected=True,
            screen_facing_score=screen_facing_score,
            yawn_detected=yawn_detected,
            yawn_count_last_3min=len(history.yawn_timestamps),
            eyes_closed=eyes_closed,
            eye_closure_duration_sec=eye_closure_duration,
            head_down=head_down,
            mouth_aspect_ratio=mar,
            eye_aspect_ratio=avg_ear,
            evidence=evidence,
        )


_DEFAULT_FACE_STATE_DETECTOR: FaceStateDetector | None = None


def get_default_face_state_detector() -> FaceStateDetector:
    global _DEFAULT_FACE_STATE_DETECTOR
    if _DEFAULT_FACE_STATE_DETECTOR is None:
        _DEFAULT_FACE_STATE_DETECTOR = FaceStateDetector()
    return _DEFAULT_FACE_STATE_DETECTOR


def detect_face_state_signals(
    face_landmarks: FaceLandmarks,
    history: HumanSensingHistory,
    head_pose: HeadPose | None = None,
) -> FaceStateSignals:
    return get_default_face_state_detector().detect_face_state_signals(
        face_landmarks=face_landmarks,
        history=history,
        head_pose=head_pose,
    )
