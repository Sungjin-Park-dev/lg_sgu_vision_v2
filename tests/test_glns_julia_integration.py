import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.glns_utils import (
    build_gtsp_problem,
    decode_and_validate_tour,
    parse_glns_tour,
    write_simple_gtsp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JULIA_PROJECT = PROJECT_ROOT / "scripts/julia/glns"
JULIA_WRAPPER = JULIA_PROJECT / "run_glns.jl"


def _find_julia():
    if os.environ.get("JULIA_BIN"):
        return Path(os.environ["JULIA_BIN"])
    installed = sorted(Path.home().glob(".julia/juliaup/julia-*/bin/julia"))
    if installed:
        return installed[-1]
    found = shutil.which("julia")
    return Path(found) if found else None


class GlnsJuliaIntegrationTests(unittest.TestCase):
    def test_tiny_delaunay_gtsp(self):
        julia = _find_julia()
        if julia is None:
            self.skipTest("Julia is not installed")
        ready = subprocess.run(
            [str(julia), f"--project={JULIA_PROJECT}", "--startup-file=no",
             "-e", "using GLNS"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if ready.returncode != 0:
            self.skipTest("GLNS.jl project is not instantiated")

        reps = [
            np.array([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]], dtype=float),
            np.array([[0.2, 0, 0, 0, 0, 0], [1.2, 0, 0, 0, 0, 0]], dtype=float),
            np.array([[0.4, 0, 0, 0, 0, 0], [1.4, 0, 0, 0, 0, 0]], dtype=float),
            np.array([[0.6, 0, 0, 0, 0, 0], [1.6, 0, 0, 0, 0, 0]], dtype=float),
        ]
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        problem = build_gtsp_problem(np.arange(4), reps, edges, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            instance = Path(tmpdir) / "tiny.gtsp"
            output = Path(tmpdir) / "tour.txt"
            write_simple_gtsp(instance, problem)
            result = subprocess.run(
                [str(julia), f"--project={JULIA_PROJECT}", "--startup-file=no",
                 str(JULIA_WRAPPER), str(instance), str(output), "fast", "30", "42"],
                capture_output=True, text=True, timeout=60, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            decoded = decode_and_validate_tour(parse_glns_tour(output), problem)

        np.testing.assert_array_equal(decoded["viewpoint_order"], np.arange(4))
        self.assertLess(decoded["cost"], problem["forbidden_cost"])


if __name__ == "__main__":
    unittest.main()
