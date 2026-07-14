from __future__ import annotations

from dataclasses import dataclass

from .contracts import AOIPrediction, FaceLandmarks, FaceStateSignals, GazePrediction, HeadPose, HumanSensingHistory, LearningState
from .face_state_detector import detect_face_state_signals, get_default_face_state_detector
from .utils import clamp


@dataclass(slots=True)
class LearningStateConfig:
    repeated_attention_duration_sec: float = 8.0
    repeated_attention_count_threshold: int = 3
    fatigue_score_threshold: float = 0.58


class LearningStateAggregator:
    def __init__(self, config: LearningStateConfig | None = None):
        self.config = config or LearningStateConfig()

    def _repeated_attention_to_same_aoi(
        self,
        history: HumanSensingHistory,
        current_gaze: GazePrediction | AOIPrediction | None,
    ) -> bool:
        if current_gaze is None:
            return False
        current_aoi = getattr(current_gaze, "predicted_aoi_id", None)
        if current_aoi and current_gaze.stable_duration_sec >= self.config.repeated_attention_duration_sec:
            return True
        if not current_aoi:
            return False
        recent_matches = [event for event in history.gaze_events if event.predicted_aoi_id == current_aoi and event.confidence >= 0.5]
        return len(recent_matches) >= self.config.repeated_attention_count_threshold

    def aggregate(
        self,
        face_state: FaceStateSignals,
        history: HumanSensingHistory,
        gaze_prediction: GazePrediction | AOIPrediction | None = None,
    ) -> LearningState:
        if gaze_prediction is not None:
            history.record_gaze(gaze_prediction)
            history.prune(gaze_prediction.timestamp)

        repeated_attention = self._repeated_attention_to_same_aoi(history, gaze_prediction)
        stable_attention_sec = 0.0 if gaze_prediction is None else gaze_prediction.stable_duration_sec
        yawn_component = min(face_state.yawn_count_last_3min / 3.0, 1.0)
        eye_component = min(face_state.eye_closure_duration_sec / 1.5, 1.0)
        posture_component = 1.0 if face_state.head_down else 0.0
        screen_component = 1.0 - face_state.screen_facing_score
        repeated_component = 0.6 if repeated_attention else 0.0

        fatigue_score = clamp(
            0.30 * yawn_component
            + 0.25 * eye_component
            + 0.20 * posture_component
            + 0.15 * screen_component
            + 0.10 * repeated_component,
            0.0,
            1.0,
        )

        possible_review_needed = repeated_attention or fatigue_score >= self.config.fatigue_score_threshold
        evidence = list(face_state.evidence)
        evidence.append(f"fatigue_signal_score={fatigue_score:.2f}")
        if repeated_attention:
            evidence.append("repeated_attention_to_same_aoi")
        if possible_review_needed:
            evidence.append("possible_review_needed")

        return LearningState(
            timestamp=face_state.timestamp,
            face_detected=face_state.face_detected,
            screen_facing_score=face_state.screen_facing_score,
            yawn_detected=face_state.yawn_detected,
            yawn_count_last_3min=face_state.yawn_count_last_3min,
            eyes_closed=face_state.eyes_closed,
            eye_closure_duration_sec=face_state.eye_closure_duration_sec,
            head_down=face_state.head_down,
            fatigue_signal_score=fatigue_score,
            possible_review_needed=possible_review_needed,
            stable_attention_sec=stable_attention_sec,
            repeated_attention_to_same_aoi=repeated_attention,
            evidence=evidence,
        )


_DEFAULT_LEARNING_STATE_AGGREGATOR: LearningStateAggregator | None = None


def get_default_learning_state_aggregator() -> LearningStateAggregator:
    global _DEFAULT_LEARNING_STATE_AGGREGATOR
    if _DEFAULT_LEARNING_STATE_AGGREGATOR is None:
        _DEFAULT_LEARNING_STATE_AGGREGATOR = LearningStateAggregator()
    return _DEFAULT_LEARNING_STATE_AGGREGATOR


def detect_learning_state(
    frame,
    face_landmarks: FaceLandmarks,
    history: HumanSensingHistory,
    head_pose: HeadPose | None = None,
    gaze_prediction: GazePrediction | AOIPrediction | None = None,
) -> LearningState:
    face_state = get_default_face_state_detector().detect_face_state_signals(
        face_landmarks=face_landmarks,
        history=history,
        head_pose=head_pose,
    )
    return get_default_learning_state_aggregator().aggregate(
        face_state=face_state,
        history=history,
        gaze_prediction=gaze_prediction,
    )
