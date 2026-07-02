import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.glns_utils import (
    build_gtsp_problem,
    decode_and_validate_tour,
    effective_candidate_cap,
    expand_edges_by_hops,
    find_hamiltonian_open_path,
    induce_adjacency,
    parse_glns_tour,
    periodic_joint_delta,
    prune_candidate_sets,
    read_result_hdf5,
    write_result_hdf5,
    write_simple_gtsp,
    unwrap_joint_path,
)


class GlnsUtilsTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        self.reps = [
            np.array([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]], dtype=float),
            np.array([[0.2, 0, 0, 0, 0, 0], [1.2, 0, 0, 0, 0, 0]], dtype=float),
            np.array([[0.4, 0, 0, 0, 0, 0], [1.4, 0, 0, 0, 0, 0]], dtype=float),
            np.array([[0.6, 0, 0, 0, 0, 0], [1.6, 0, 0, 0, 0, 0]], dtype=float),
        ]

    def test_induced_components_after_drop(self):
        induced, labels, components = induce_adjacency(
            self.edges, np.array([True, False, True, True]),
        )
        np.testing.assert_array_equal(induced, np.array([[2, 3]], dtype=np.int32))
        self.assertEqual(labels[1], -1)
        self.assertEqual([c.tolist() for c in components], [[0], [2, 3]])

    def test_expand_edges_by_hops(self):
        # Path 0-1-2-3: 1-hop is unchanged; 2-hop adds (0,2) and (1,3).
        np.testing.assert_array_equal(
            expand_edges_by_hops(self.edges, 4, 1), self.edges,
        )
        two_hop = {tuple(e) for e in expand_edges_by_hops(self.edges, 4, 2).tolist()}
        self.assertEqual(
            two_hop, {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)},
        )
        # 3-hop closes the whole path into a clique.
        three_hop = {tuple(e) for e in expand_edges_by_hops(self.edges, 4, 3).tolist()}
        self.assertEqual(three_hop, {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)})

    def test_hamiltonian_open_path_feasibility(self):
        status, witness = find_hamiltonian_open_path(np.arange(4), self.edges)
        self.assertIn(status, ("optimal", "feasible"))
        self.assertEqual(set(witness.tolist()), {0, 1, 2, 3})

        star = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int32)
        status, witness = find_hamiltonian_open_path(np.arange(5), star)
        self.assertEqual(status, "infeasible")
        self.assertIsNone(witness)

    def test_gtsp_cost_is_lexicographic_and_non_edges_forbidden(self):
        problem = build_gtsp_problem(
            np.arange(4), self.reps, self.edges, reconfig_threshold_rad=0.5,
        )
        costs = problem["costs"]
        self.assertGreater(problem["reconfig_unit"], 3 * 1200)
        # Three-tier lexicographic ordering: base ≫ wrist ≫ L2 sum (3*1200).
        self.assertGreater(problem["reconfig_unit_wrist"], 3 * 1200)
        self.assertGreater(problem["reconfig_unit_base"], problem["reconfig_unit_wrist"])
        self.assertEqual(problem["reconfig_unit"], problem["reconfig_unit_base"])

        # vp0 vertex 0 -> vp2 vertex 4 is not a Delaunay transition.
        self.assertEqual(costs[0, 4], problem["forbidden_cost"])
        # Zero-tilt dummy endpoints remain free.
        self.assertTrue(np.all(costs[problem["dummy_vertex"], :-1] == 0))

    def test_strict_tiers_and_dummy_tilt_cost(self):
        tilt = [np.array([0, 4]), np.array([1, 0]), np.array([0, 0]), np.array([0, 0])]
        problem = build_gtsp_problem(
            np.arange(4), self.reps, self.edges, 0.5,
            candidate_tilt_costs=tilt,
        )
        self.assertGreater(problem["tilt_unit"], problem["max_l2_sum"])
        self.assertGreater(
            problem["reconfig_unit_any"],
            problem["max_tilt_sum"] + problem["max_l2_sum"],
        )
        self.assertGreater(
            problem["reconfig_unit_base"],
            problem["max_any_sum"] + problem["max_tilt_sum"] + problem["max_l2_sum"],
        )
        self.assertGreater(problem["forbidden_cost"], problem["max_allowed_tour"])
        dummy = problem["dummy_vertex"]
        self.assertEqual(problem["costs"][dummy, 1], 4 * problem["tilt_unit"])

    def test_candidate_cap_and_deterministic_pruning(self):
        cap = effective_candidate_cap(471, 16, 256.0)
        self.assertEqual(cap, 12)
        matrix_mib = (471 * cap + 1) ** 2 * 8 / 1024 ** 2
        self.assertLessEqual(matrix_mib, 256.0)

        reps = [
            np.array([[0.0] * 6, [0.1] * 6, [1.0] * 6]),
            np.array([[0.15] * 6, [1.1] * 6]),
        ]
        metadata = []
        for count in (3, 2):
            metadata.append({
                "variant": np.array(["nominal"] + ["tilt"] * (count - 1)),
                "tilt_deg": np.array([0.0] + [5.0] * (count - 1)),
                "roll_deg": np.zeros(count),
                "tilt_azimuth_deg": np.full(count, np.nan),
                "target_position": np.zeros((count, 3)),
                "target_quaternion": np.zeros((count, 4)),
            })
        kwargs = dict(
            representatives=reps, metadata=metadata,
            edges=np.array([[0, 1]], dtype=np.int32),
            cap_by_viewpoint=np.array([2, 2]), threshold_rad=0.3,
            joint_weights=np.ones(6),
        )
        p1, m1 = prune_candidate_sets(**kwargs)
        p2, m2 = prune_candidate_sets(**kwargs)
        np.testing.assert_array_equal(p1[0], p2[0])
        self.assertEqual(m1[0]["variant"][0], "nominal")
        self.assertLessEqual(len(p1[0]), 2)

    def test_periodic_cost_and_limit_aware_path_unwrap(self):
        periods = np.array([2 * np.pi, 2 * np.pi, 0, 2 * np.pi, 2 * np.pi, 2 * np.pi])
        lower = np.array([-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi])
        upper = -lower
        q0 = np.deg2rad([0, 0, 0, 0, 156.8, 0])
        q1 = np.deg2rad([0, 0, 0, 0, -103.6, 0])
        shortest = periodic_joint_delta(q1 - q0, periods)
        self.assertAlmostEqual(abs(np.rad2deg(shortest[4])), 99.6, places=6)

        unwrapped = unwrap_joint_path(
            np.stack([q0, q1]), lower, upper, periods,
            threshold_rad=np.deg2rad(120), reference_joints=np.zeros(6),
        )
        self.assertAlmostEqual(
            abs(np.rad2deg(unwrapped[1, 4] - unwrapped[0, 4])), 99.6, places=6,
        )
        self.assertTrue(np.all(unwrapped >= lower - 1e-9))
        self.assertTrue(np.all(unwrapped <= upper + 1e-9))
        np.testing.assert_allclose(
            periodic_joint_delta(unwrapped - np.stack([q0, q1]), periods), 0.0,
            atol=1e-10,
        )

        reps = [q0[None, :], q1[None, :]]
        problem = build_gtsp_problem(
            np.arange(2), reps, np.array([[0, 1]], dtype=np.int32),
            reconfig_threshold_rad=np.deg2rad(120), joint_periods=periods,
        )
        self.assertLess(problem["costs"][0, 1], problem["reconfig_unit_any"])

    def test_pruning_reuses_edge_matrix_for_both_endpoint_scores(self):
        reps = [
            np.array([[0.0] * 6, [2.0] * 6]),
            np.array([[2.1] * 6, [4.0] * 6]),
        ]
        metadata = [{
            "variant": np.array(["tilt", "tilt"]),
            "tilt_deg": np.array([5.0, 5.0]),
        } for _ in reps]
        pruned, _ = prune_candidate_sets(
            reps, metadata, np.array([[0, 1]], dtype=np.int32),
            cap_by_viewpoint=np.array([1, 1]), threshold_rad=0.3,
            joint_weights=np.ones(6), reference_joints=np.zeros(6),
        )
        np.testing.assert_array_equal(pruned[0][0], np.full(6, 2.0))
        np.testing.assert_array_equal(pruned[1][0], np.full(6, 2.1))

    def test_decode_open_tour_and_reject_forbidden_transition(self):
        problem = build_gtsp_problem(
            np.arange(4), self.reps, self.edges, reconfig_threshold_rad=0.5,
        )
        dummy = problem["dummy_vertex"]
        decoded = decode_and_validate_tour(
            np.array([dummy, 0, 2, 4, 6], dtype=np.int32), problem,
        )
        np.testing.assert_array_equal(decoded["viewpoint_order"], np.arange(4))
        np.testing.assert_array_equal(decoded["candidate_order"], np.zeros(4, dtype=np.int32))

        with self.assertRaisesRegex(ValueError, "forbidden"):
            decode_and_validate_tour(
                np.array([dummy, 0, 4, 2, 6], dtype=np.int32), problem,
            )

    def test_text_io_and_result_hdf5_round_trip(self):
        problem = build_gtsp_problem(
            np.arange(4), self.reps, self.edges, reconfig_threshold_rad=0.5,
        )
        component = {
            "members": np.arange(4), "status": "solved", "reason": "",
            "feasibility_witness": np.arange(4), "solver_cost": 600,
            "reconfig_unit": problem["reconfig_unit"],
            "forbidden_cost": problem["forbidden_cost"], "joint_cost_scale": 1000,
            "reconfig_unit_base": problem["reconfig_unit_base"],
            "reconfig_unit_wrist": problem["reconfig_unit_wrist"],
            "num_reconfigurations": 0, "num_reconfigurations_base": 0,
            "num_reconfigurations_wrist": 0,
            "solver_seconds": 0.1, "matrix_mib": 0.01,
            "viewpoint_order": np.arange(4),
            "selected_candidate_index": np.zeros(4, dtype=np.int32),
            "selected_joints": np.stack([r[0] for r in self.reps]),
            "edge_linf_rad": np.full(3, 0.2), "edge_l2_rad": np.full(3, 0.2),
            "edge_linf_base_rad": np.full(3, 0.2),
            "edge_linf_wrist_rad": np.zeros(3),
            "is_reconfiguration": np.zeros(3, dtype=bool),
            "is_reconfiguration_base": np.zeros(3, dtype=bool),
            "is_reconfiguration_wrist": np.zeros(3, dtype=bool),
            "selected_pose_variant": np.array(["nominal", "roll", "tilt", "nominal"]),
            "selected_roll_deg": np.array([0.0, 30.0, 0.0, 0.0]),
            "selected_tilt_deg": np.array([0.0, 0.0, 5.0, 0.0]),
            "selected_tilt_azimuth_deg": np.array([np.nan, np.nan, 45.0, np.nan]),
            "selected_target_position": np.zeros((4, 3)),
            "selected_target_quaternion": np.zeros((4, 4)),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_simple_gtsp(root / "tiny.gtsp", problem)
            (root / "tour.txt").write_text(
                f"Tour Cost : 600\nTour : [{problem['dummy_vertex'] + 1}, 1, 3, 5, 7]",
                encoding="utf-8",
            )
            parsed = parse_glns_tour(root / "tour.txt")
            self.assertEqual(parsed[0], problem["dummy_vertex"])

            result_path = root / "result.h5"
            write_result_hdf5(
                result_path,
                {"object": "synthetic", "object_position": [0, 0, 0]},
                np.ones(4, dtype=bool), np.full(4, 2), self.edges,
                np.zeros(4, dtype=np.int32), [component],
                candidate_counts_raw=np.full(4, 5),
            )
            loaded = read_result_hdf5(result_path)
            self.assertEqual(loaded["components"][0]["status"], "solved")
            np.testing.assert_array_equal(
                loaded["components"][0]["viewpoint_order"], np.arange(4),
            )
            # joint-differentiated tier fields round-trip through the HDF5 schema.
            np.testing.assert_array_equal(
                loaded["components"][0]["is_reconfiguration_base"],
                np.zeros(3, dtype=bool),
            )
            self.assertEqual(
                int(loaded["components"][0]["attrs"]["num_reconfigurations_wrist"]), 0,
            )
            np.testing.assert_array_equal(loaded["candidate_counts_raw"], np.full(4, 5))
            np.testing.assert_array_equal(
                loaded["components"][0]["selected_tilt_deg"],
                np.array([0.0, 0.0, 5.0, 0.0]),
            )

            # A v1 file has no raw-count dataset; the reader falls back to pruned counts.
            import h5py
            with h5py.File(result_path, "a") as f:
                f.attrs["format_version"] = 1
                del f["input/candidate_counts_raw"]
            loaded_v1 = read_result_hdf5(result_path)
            np.testing.assert_array_equal(
                loaded_v1["candidate_counts_raw"], loaded_v1["candidate_counts"],
            )


if __name__ == "__main__":
    unittest.main()
