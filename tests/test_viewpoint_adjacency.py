import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from core.generate_viewpoints import (
    build_local_delaunay_adjacency,
    save_viewpoints_hdf5,
)
from apps.viewpoint_studio import load_viewpoint_h5


class LocalDelaunayAdjacencyTests(unittest.TestCase):
    def test_planar_grid_is_connected_and_canonical(self):
        xy = np.array([(x, y) for y in range(4) for x in range(4)], dtype=np.float64) * 0.01
        points = np.column_stack([xy, np.zeros(len(xy))])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

        result = build_local_delaunay_adjacency(points, normals, k_neighbors=8)
        edges = result["edges"]

        self.assertGreater(len(edges), 0)
        self.assertEqual(result["stats"]["num_components"], 1)
        self.assertEqual(result["stats"]["num_isolated"], 0)
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        self.assertEqual(len(edges), len(np.unique(edges, axis=0)))

        repeat = build_local_delaunay_adjacency(points, normals, k_neighbors=8)
        np.testing.assert_array_equal(edges, repeat["edges"])

    def test_opposite_surface_normals_block_cross_surface_edges(self):
        xy = np.array([(x, y) for y in range(3) for x in range(3)], dtype=np.float64) * 0.01
        upper = np.column_stack([xy, np.zeros(len(xy))])
        lower = np.column_stack([xy, np.full(len(xy), 0.002)])
        points = np.vstack([upper, lower])
        normals = np.vstack([
            np.tile([0.0, 0.0, 1.0], (len(upper), 1)),
            np.tile([0.0, 0.0, -1.0], (len(lower), 1)),
        ])

        result = build_local_delaunay_adjacency(
            points, normals, k_neighbors=12, max_normal_angle_deg=75.0,
        )
        split = len(upper)
        cross = ((result["edges"][:, 0] < split)
                 & (result["edges"][:, 1] >= split))

        self.assertFalse(np.any(cross))
        self.assertEqual(result["stats"]["num_components"], 2)
        self.assertEqual(result["stats"]["num_isolated"], 0)

    def test_collinear_points_use_one_dimensional_adjacency(self):
        points = np.column_stack([
            np.arange(7, dtype=np.float64) * 0.01,
            np.zeros(7),
            np.zeros(7),
        ])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

        result = build_local_delaunay_adjacency(points, normals, k_neighbors=4)
        expected = np.array([(i, i + 1) for i in range(6)], dtype=np.int32)

        np.testing.assert_array_equal(result["edges"], expected)
        self.assertEqual(result["stats"]["num_components"], 1)

    def test_hdf5_round_trip(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.01, 0.01, 0.0],
        ])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        adjacency = build_local_delaunay_adjacency(points, normals, k_neighbors=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "viewpoints.h5"
            save_viewpoints_hdf5(points, normals, output, adjacency=adjacency)

            with h5py.File(output, "r") as f:
                group = f["viewpoints/adjacency"]
                np.testing.assert_array_equal(group["edges"][:], adjacency["edges"])
                np.testing.assert_array_equal(
                    group["component_id"][:], adjacency["component_id"],
                )
                self.assertEqual(group.attrs["method"], "local_tangent_delaunay")
                self.assertEqual(group.attrs["num_edges"], len(adjacency["edges"]))

            loaded = load_viewpoint_h5(output)
            np.testing.assert_array_equal(
                loaded["adjacency"]["edges"], adjacency["edges"],
            )

    def test_studio_loader_accepts_legacy_hdf5_without_adjacency(self):
        points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "legacy.h5"
            save_viewpoints_hdf5(points, normals, output)
            loaded = load_viewpoint_h5(output)

        self.assertIsNone(loaded["adjacency"])


if __name__ == "__main__":
    unittest.main()
