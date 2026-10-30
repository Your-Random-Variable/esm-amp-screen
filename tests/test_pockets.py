"""
Unit tests for 7-Ray PSP LIGSITE Cavity Detection and Multi-Pocket Blind Docking.
"""

import unittest
from pathlib import Path
import tempfile
import gemmi

from esm_amp_screen.receptor_prep import detect_surface_cavities, calculate_grid_box
from esm_amp_screen.reporter import generate_pockets_pymol_script


class TestPocketDetection(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.mdm2_pdb = self.root / "benchmarks" / "1ycr.pdb"
        self.cypa_pdb = self.root / "benchmarks" / "1awr.pdb"
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mdm2_five_pockets(self):
        """Verify 5 distinct pockets are discovered on MDM2 with Pocket #1 as primary cleft."""
        pockets = detect_surface_cavities(self.mdm2_pdb, max_pockets=5, min_pocket_distance=8.0)
        self.assertEqual(len(pockets), 5, f"Expected 5 pockets on MDM2, got {len(pockets)}")

        p1 = pockets[0]
        self.assertEqual(p1["pocket_id"], 1)
        self.assertGreaterEqual(p1["druggability_score"], 75.0, "Pocket #1 should have high druggability score")
        self.assertIn("High Druggability", p1["druggability_class"])
        self.assertGreaterEqual(p1["volume_angstrom3"], 500.0)
        self.assertGreaterEqual(p1["hydrophobic_ratio"], 0.35)

        # Verify canonical p53 binding residues are in Pocket #1
        lining_str = " ".join(p1["contact_residues_formatted"])
        for canonical_res in ["LEU", "PHE", "TYR"]:
            self.assertIn(canonical_res, lining_str)

        # Verify spatial distinctiveness between centers (>= 8.0 A)
        import numpy as np
        for i in range(len(pockets)):
            ci = np.array([pockets[i]["center_x"], pockets[i]["center_y"], pockets[i]["center_z"]])
            for j in range(i + 1, len(pockets)):
                cj = np.array([pockets[j]["center_x"], pockets[j]["center_y"], pockets[j]["center_z"]])
                dist = np.linalg.norm(ci - cj)
                self.assertGreaterEqual(dist, 7.5, f"Pockets {i+1} and {j+1} too close: {dist:.1f} A")

    def test_cypa_five_pockets(self):
        """Verify 5 distinct pockets on Cyclophilin A."""
        pockets = detect_surface_cavities(self.cypa_pdb, max_pockets=5, min_pocket_distance=8.0)
        self.assertEqual(len(pockets), 5)
        self.assertGreaterEqual(pockets[0]["druggability_score"], 70.0)

    def test_calculate_grid_box_auto_pocket(self):
        """Verify calculate_grid_box attaches detected_pockets list."""
        grid = calculate_grid_box(self.mdm2_pdb, auto_pocket=True, max_pockets=5)
        self.assertIn("detected_pockets", grid)
        self.assertEqual(len(grid["detected_pockets"]), 5)
        self.assertAlmostEqual(grid["center_x"], grid["detected_pockets"][0]["center_x"], places=2)

    def test_generate_pockets_pymol_script(self):
        """Verify PyMOL script generation for 5 pockets."""
        pockets = detect_surface_cavities(self.mdm2_pdb, max_pockets=5)
        pml_path = self.test_dir / "view_pockets.pml"
        generate_pockets_pymol_script(self.mdm2_pdb, pockets, pml_path)
        self.assertTrue(pml_path.exists())
        content = pml_path.read_text()
        self.assertIn("pocket_1_res", content)
        self.assertIn("p1_center", content)
        self.assertIn("firebrick", content)
        self.assertIn("forest", content)


if __name__ == "__main__":
    unittest.main()
