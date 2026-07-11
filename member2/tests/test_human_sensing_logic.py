from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.human_sensing.calibration import CalibrationSession
from modules.human_sensing.contracts import AOIPrediction, FaceStateSignals, GazeFeatures, GazePrediction, HumanSensingHistory
from modules.human_sensing.gaze_estimator import map_gaze_to_aoi, normalize_head_pose_angles
from modules.human_sensing.learning_state_aggregator import LearningStateAggregator
from modules.human_sensing.utils import canonicalize_grid_name, grid_cell_bbox


class HumanSensingLogicTests(unittest.TestCase):
    def test_grid_alias_is_normalized(self) -> None:
        self.assertEqual(canonicalize_grid_name("left_top"), "top_left")
        self.assertEqual(canonicalize_grid_name("middle_right"), "middle_right")

    def test_grid_cell_bbox_accepts_legacy_and_row_first_names(self) -> None:
        self.assertEqual(grid_cell_bbox("middle_right").to_list(), grid_cell_bbox("right_middle").to_list())

    def test_head_pose_pitch_flip_is_folded_near_zero(self) -> None:
        yaw, pitch, roll = normalize_head_pose_angles(-19.4, 179.9, -3.4)
        self.assertAlmostEqual(yaw, -19.4)
        self.assertAlmostEqual(pitch, -0.1)
        self.assertAlmostEqual(roll, -3.4)

    def test_map_gaze_to_aoi_prefers_matching_side(self) -> None:
        gaze = GazePrediction(
            timestamp=1.0,
            slide_id=5,
            gaze_grid="middle_right",
            confidence=0.8,
            stable_duration_sec=2.3,
        )
        aois = [
            {"aoi_id": "left_text", "bbox": [0.05, 0.18, 0.48, 0.78], "type": "text"},
            {"aoi_id": "right_figure", "bbox": [0.55, 0.18, 0.95, 0.78], "type": "figure"},
        ]
        prediction = map_gaze_to_aoi(gaze, aois)
        self.assertEqual(prediction.predicted_aoi_id, "right_figure")
        self.assertGreater(prediction.confidence, 0.0)

    def test_member1_gaze_payload_is_accepted_by_member2_mapping(self) -> None:
        member1_payload = {
            "slide_id": 2,
            "aois": [
                {
                    "aoi_id": "left_block",
                    "bbox": [0.05, 0.35, 0.48, 0.85],
                    "type": "text",
                    "source": "rule",
                    "include_in_learning": True,
                },
                {
                    "aoi_id": "right_visual_region",
                    "bbox": [0.50, 0.43, 0.96, 0.80],
                    "type": "figure",
                    "source": "rule",
                    "include_in_learning": True,
                },
            ],
        }
        gaze = GazePrediction(
            timestamp=1.0,
            slide_id=member1_payload["slide_id"],
            gaze_grid="middle_right",
            confidence=0.82,
            stable_duration_sec=1.4,
        )
        prediction = map_gaze_to_aoi(gaze, member1_payload["aois"])
        self.assertEqual(prediction.slide_id, 2)
        self.assertEqual(prediction.predicted_aoi_id, "right_visual_region")
        self.assertGreater(prediction.confidence, 0.0)

    def test_calibration_session_builds_centroids(self) -> None:
        session = CalibrationSession(user_id="demo")
        grids = [
            "left_top",
            "top_center",
            "top_right",
            "middle_left",
            "center_middle",
            "middle_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ]
        for index, grid in enumerate(grids):
            features = GazeFeatures(
                timestamp=float(index),
                yaw=float(index),
                pitch=float(index) / 2.0,
                left_iris_offset_x=0.01 * index,
                right_iris_offset_x=0.02 * index,
                face_detected=True,
            )
            session.record_sample(grid, features)
        profile = session.build_profile()
        self.assertIn("middle_right", profile.grid_centroids)
        self.assertEqual(profile.user_id, "demo")

    def test_learning_state_flags_long_attention(self) -> None:
        history = HumanSensingHistory()
        aggregator = LearningStateAggregator()
        face_state = FaceStateSignals(
            timestamp=10.0,
            face_detected=True,
            screen_facing_score=0.9,
            yawn_detected=False,
            yawn_count_last_3min=0,
            eyes_closed=False,
            eye_closure_duration_sec=0.0,
            head_down=False,
            mouth_aspect_ratio=0.15,
            eye_aspect_ratio=0.25,
        )
        gaze = AOIPrediction(
            timestamp=10.0,
            slide_id=5,
            gaze_grid="middle_right",
            predicted_aoi_id="right_figure",
            confidence=0.82,
            stable_duration_sec=9.2,
        )
        learning_state = aggregator.aggregate(face_state, history, gaze)
        self.assertTrue(learning_state.repeated_attention_to_same_aoi)
        self.assertTrue(learning_state.possible_review_needed)


if __name__ == "__main__":
    unittest.main()
