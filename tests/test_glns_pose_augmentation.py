import unittest

import numpy as np

from core.solve_glns_path import _build_pose_variants


class PoseAugmentationTests(unittest.TestCase):
    def test_default_pose_variants_preserve_surface_and_working_distance(self):
        pose = np.eye(4, dtype=np.float64)[None, ...]
        pose[0, :3, 3] = [0.2, -0.1, 0.5]
        wd_m = 0.3
        targets = _build_pose_variants(
            pose, wd_m, roll_augment=True, roll_step_deg=30.0,
            tilt_augment=True, tilt_angles_deg=(5.0, 10.0), tilt_azimuths=8,
        )

        self.assertEqual(len(targets["position"]), 28)
        self.assertEqual(list(targets["variant"]).count("nominal"), 1)
        self.assertEqual(list(targets["variant"]).count("roll"), 11)
        self.assertEqual(list(targets["variant"]).count("tilt"), 16)
        self.assertFalse(np.any((targets["roll_deg"] != 0.0) & (targets["tilt_deg"] != 0.0)))

        surface = pose[0, :3, 3] + pose[0, :3, 2] * wd_m
        reconstructed = targets["position"] + targets["rotation"][:, :, 2] * wd_m
        np.testing.assert_allclose(
            reconstructed, np.repeat(surface[None, :], len(reconstructed), axis=0), atol=1e-10,
        )
        distances = np.linalg.norm(surface[None, :] - targets["position"], axis=1)
        np.testing.assert_allclose(distances, wd_m, atol=1e-10)

        tilt_mask = targets["variant"] == "tilt"
        angles = np.rad2deg(np.arccos(
            np.clip(targets["rotation"][tilt_mask, 2, 2], -1, 1)))
        np.testing.assert_allclose(angles, targets["tilt_deg"][tilt_mask], atol=1e-8)
        np.testing.assert_array_equal(
            np.unique(targets["tilt_azimuth_deg"][tilt_mask]),
            np.arange(0.0, 360.0, 45.0),
        )


if __name__ == "__main__":
    unittest.main()
