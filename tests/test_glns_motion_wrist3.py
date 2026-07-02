import unittest
from unittest import mock
import tempfile
from pathlib import Path

import numpy as np

from core import verify_glns_trajectory as verifier


class GlnsMotionWrist3Tests(unittest.TestCase):
    def test_component_order_ignores_home_pose(self):
        endpoints = [
            (np.full(6, 10.0), np.full(6, 11.0)),
            (np.full(6, 0.0), np.full(6, 1.0)),
        ]
        near_first = verifier._choose_order(endpoints, np.zeros(6))
        near_second = verifier._choose_order(endpoints, np.full(6, 100.0))
        self.assertEqual(near_first, near_second)

    def test_home_transitions_are_planned_and_saved_separately(self):
        home = np.zeros(6)
        scan = np.array([[1.0] * 6, [2.0] * 6])
        segments = [(np.stack([home, scan[0]]), "direct"),
                    (np.stack([scan[-1], home]), "via-home")]
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            verifier, "_plan_seams_batched", return_value=segments,
        ) as plan, mock.patch.object(
            verifier, "_resample_seam",
            side_effect=lambda a, b, *_args, **_kwargs: (
                np.stack([a, b]), np.ones(2, dtype=bool)),
        ), mock.patch.object(
            verifier, "_collision_gate_and_save",
            side_effect=lambda traj, mask, **kwargs: {
                "collision_free": True, "n_waypoints": len(traj),
                "total_time": 1.0, "csv": str(kwargs["out_csv"]),
            },
        ):
            results = verifier._plan_home_transitions(
                scan, home, robot_cfg={}, world_config=object(), wd_m=0.3,
                spacing=0.01, reconfig_rad=0.5, enable_via_ladder=True,
                motion_planner=object(), out_dir=Path(tmpdir),
            )

        self.assertTrue(all(item["ok"] for item in results))
        pairs = plan.call_args.args[0]
        np.testing.assert_array_equal(pairs[0][0], home)
        np.testing.assert_array_equal(pairs[0][1], scan[0])
        np.testing.assert_array_equal(pairs[1][0], scan[-1])
        np.testing.assert_array_equal(pairs[1][1], home)
        self.assertTrue(results[0]["gate"]["csv"].endswith("home_to_start.csv"))
        self.assertTrue(results[1]["gate"]["csv"].endswith("end_to_home.csv"))

    def test_single_home_transition_plans_only_requested_leg(self):
        home = np.zeros(6)
        scan = np.array([[1.0] * 6, [2.0] * 6])
        segment = [(np.stack([home, scan[0]]), "direct")]
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            verifier, "_plan_seams_batched", return_value=segment,
        ) as plan, mock.patch.object(
            verifier, "_resample_seam",
            side_effect=lambda a, b, *_args, **_kwargs: (
                np.stack([a, b]), np.ones(2, dtype=bool)),
        ), mock.patch.object(
            verifier, "_collision_gate_and_save",
            return_value={"collision_free": True, "n_waypoints": 2,
                          "total_time": 1.0, "csv": "home_to_start.csv"},
        ):
            results = verifier._plan_home_transitions(
                scan, home, robot_cfg={}, world_config=object(), wd_m=0.3,
                spacing=0.01, reconfig_rad=0.5, enable_via_ladder=True,
                motion_planner=object(), out_dir=Path(tmpdir), transitions="approach",
            )

        self.assertEqual(len(results), 1)
        pairs = plan.call_args.args[0]
        self.assertEqual(len(pairs), 1)
        np.testing.assert_array_equal(pairs[0][0], home)
        np.testing.assert_array_equal(pairs[0][1], scan[0])

    def test_component_transit_preserves_selected_wrist3(self):
        selected = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
            [1.0, 0.0, 0.0, 0.0, 0.0, -0.7],
        ])
        planned = selected.copy()
        component = {
            "selected_joints": selected,
            "viewpoint_order": np.array([10, 11]),
            "is_reconfiguration": np.array([True]),
        }
        shared_planner = object()

        with mock.patch.object(
            verifier.PT, "find_colliding_interpolation_edges", return_value=np.array([], dtype=int),
        ), mock.patch.object(
            verifier.PT, "plan_reconfig_transits",
            return_value=({0: planned.copy()}, [{"idx": 0, "success": True, "route": "direct"}]),
        ) as plan, mock.patch.object(
            verifier.PT, "interpolate_and_resample",
            return_value=(selected.copy(), np.array([False, False]), [],
                          {"runs": [(0, 1, 2)], "kept": (0, 1, 2)}),
        ) as interpolate:
            result = verifier._plan_and_resample_component(
                component, robot_cfg={}, world_config=object(), reconfig_rad=0.5,
                wd_m=0.3, spacing=0.01, motion_planner=shared_planner,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(plan.call_args.kwargs["lock_wrist3"])
        self.assertIs(plan.call_args.kwargs["motion_planner"], shared_planner)
        np.testing.assert_array_equal(
            interpolate.call_args.args[1][0][:, -1], np.array([0.6, -0.7]),
        )

    def test_colliding_small_jump_uses_motion_planning_before_drop(self):
        selected = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        component = {
            "selected_joints": selected,
            "viewpoint_order": np.array([20, 21, 22]),
            "is_reconfiguration": np.array([False, False]),
        }
        fallback = np.stack([selected[0], selected[1]])

        with mock.patch.object(
            verifier.PT, "find_colliding_interpolation_edges",
            return_value=np.array([0], dtype=int),
        ) as detect, mock.patch.object(
            verifier.PT, "plan_reconfig_transits",
            return_value=({0: fallback}, [{"idx": 0, "success": True, "route": "direct"}]),
        ) as plan, mock.patch.object(
            verifier.PT, "interpolate_and_resample",
            return_value=(selected.copy(), np.array([True, True, False]), [],
                          {"runs": [(0, 2, 3)], "kept": (0, 2, 3)}),
        ) as interpolate:
            result = verifier._plan_and_resample_component(
                component, robot_cfg={}, world_config=object(), reconfig_rad=0.5,
                wd_m=0.3, spacing=0.01, motion_planner=object(),
            )

        np.testing.assert_array_equal(detect.call_args.args[1], np.array([0, 1]))
        np.testing.assert_array_equal(plan.call_args.args[1], np.array([0]))
        self.assertIn(0, interpolate.call_args.args[1])
        self.assertEqual(result["reconfig_req"], 0)
        self.assertEqual(result["transit_req"], 1)
        self.assertEqual(result["collision_fallback_req"], 1)
        self.assertEqual(result["collision_fallback_ok"], 1)
        self.assertEqual(result["dropped"], [])

    def test_interpolation_collision_detection_batches_edges(self):
        selected = np.zeros((4, 6), dtype=float)
        selected[1:, 0] = [0.1, 0.2, 0.3]

        # The first dense segment is free and the second contains one collision.
        with mock.patch.object(
            verifier.PT, "densify_for_collision_check",
            side_effect=lambda segment: np.linspace(segment[0], segment[1], 3),
        ), mock.patch.object(
            verifier.PT, "batch_collision_check",
            return_value=(np.array([False, False, False, False, True, False]), 1),
        ) as collision_check:
            found = verifier.PT.find_colliding_interpolation_edges(
                selected, np.array([0, 2]), robot_cfg={}, world_scene=object(),
            )

        np.testing.assert_array_equal(found, np.array([2]))
        self.assertEqual(len(collision_check.call_args.args[0]), 6)

    def test_seam_transit_preserves_planned_wrist3(self):
        q0 = np.array([0, 0, 0, 0, 0, 0.8], dtype=float)
        q1 = np.array([1, 0, 0, 0, 0, -0.9], dtype=float)
        planned = np.stack([q0, q1])
        shared_planner = object()
        with mock.patch.object(
            verifier.PT, "plan_reconfig_transits",
            return_value=({0: planned.copy()}, [{"idx": 0, "success": True, "route": "via-home"}]),
        ) as plan:
            result = verifier._plan_seams_batched(
                [(q0, q1)], robot_cfg={}, world_config=object(), wd_m=0.3,
                motion_planner=shared_planner,
            )

        self.assertFalse(plan.call_args.kwargs["lock_wrist3"])
        self.assertIs(plan.call_args.kwargs["motion_planner"], shared_planner)
        np.testing.assert_array_equal(result[0][0][:, -1], np.array([0.8, -0.9]))


if __name__ == "__main__":
    unittest.main()
