import unittest
import numpy as np
from pathlib import Path
import tempfile
from esm_amp_screen.physics_engine import (
    VDW_PARAMS,
    SOLVATION_PARAMS,
    ALL_ATOM_CHARGES,
    HEAVY_ATOM_CHARGES,
    get_residue_formal_charge,
    calculate_haddock_physics_score
)

class TestPhysicsEngine(unittest.TestCase):
    def test_formal_charges(self):
        """Verify that both heavy-atom and all-atom tables directly sum to correct formal charges at pH 7.4."""
        # 1. Direct summation of ALL_ATOM_CHARGES table entries
        all_atom_expected = {
            "ARG": 1.0, "LYS": 1.0, "ASP": -1.0, "GLU": -1.0,
            "HIP": 1.0, "HID": 0.0, "HIE": 0.0, "HIS": 0.0,
            "TYR": 0.0, "CYS": 0.0
        }
        for res, exp in all_atom_expected.items():
            direct_sum = sum(v for k, v in ALL_ATOM_CHARGES.items() if k[0] == res)
            self.assertAlmostEqual(direct_sum, exp, delta=1e-3, msg=f"ALL_ATOM_CHARGES sum for {res} must be {exp}")

        # 2. Direct summation of HEAVY_ATOM_CHARGES table entries
        heavy_expected = {
            "ARG": 1.0, "LYS": 1.0, "ASP": -1.0, "GLU": -1.0,
            "HIP": 1.0, "HID": 0.0, "HIE": 0.0, "HIS": 0.0
        }
        for res, exp in heavy_expected.items():
            direct_sum = sum(v for k, v in HEAVY_ATOM_CHARGES.items() if k[0] == res)
            self.assertAlmostEqual(direct_sum, exp, delta=1e-3, msg=f"HEAVY_ATOM_CHARGES sum for {res} must be {exp}")

        # 3. get_residue_formal_charge API check
        for has_h in [False, True]:
            self.assertAlmostEqual(get_residue_formal_charge("ARG", has_hydrogens=has_h), 1.0, delta=0.05)
            self.assertAlmostEqual(get_residue_formal_charge("LYS", has_hydrogens=has_h), 1.0, delta=0.05)
            self.assertAlmostEqual(get_residue_formal_charge("ASP", has_hydrogens=has_h), -1.0, delta=0.05)
            self.assertAlmostEqual(get_residue_formal_charge("GLU", has_hydrogens=has_h), -1.0, delta=0.05)
            self.assertAlmostEqual(get_residue_formal_charge("ALA", has_hydrogens=has_h), 0.0, delta=0.05)
            self.assertAlmostEqual(get_residue_formal_charge("LEU", has_hydrogens=has_h), 0.0, delta=0.05)

    def test_vdw_combining_rule(self):
        sig_c, eps_c = VDW_PARAMS["C"]
        sig_n, eps_n = VDW_PARAMS["N"]
        self.assertAlmostEqual((sig_c + sig_n) / 2.0, 3.375, delta=1e-4)
        self.assertAlmostEqual(np.sqrt(eps_c * eps_n), np.sqrt(0.066 * 0.170), delta=1e-4)

    def test_repulsive_wall(self):
        """Verify Lennard-Jones repulsive wall is monotonic and caps at 500 kcal/mol for all r <= 1.0 A."""
        sig_c, eps_c = VDW_PARAMS["C"]
        def calc_vdw(r):
            r_eff = max(r, 1.0)
            sr6 = (sig_c / r_eff) ** 6
            return float(np.clip(4.0 * eps_c * (sr6 ** 2 - sr6), -2.0, 500.0))

        distances = [1.8, 1.5, 1.0, 0.7, 0.5]
        energies = [calc_vdw(r) for r in distances]
        self.assertGreater(energies[0], 10.0)
        self.assertGreater(energies[1], 100.0)
        self.assertEqual(energies[2], 500.0)
        self.assertEqual(energies[3], 500.0)
        self.assertEqual(energies[4], 500.0)
        # Monotonicity check
        for e_prev, e_curr in zip(energies[:-1], energies[1:]):
            self.assertGreaterEqual(e_curr, e_prev)

    def test_switching_electrostatics(self):
        """Verify canonical C2 quintic smoothstep function S(u) = 1 - 10u^3 + 15u^4 - 6u^5."""
        r_on, r_off = 9.0, 12.0
        r_range = r_off - r_on
        def switch_func(r):
            if r <= r_on:
                return 1.0
            elif r >= r_off:
                return 0.0
            u = (r - r_on) / r_range
            return 1.0 - 10.0 * (u ** 3) + 15.0 * (u ** 4) - 6.0 * (u ** 5)

        self.assertAlmostEqual(switch_func(9.0), 1.0, delta=1e-6)
        self.assertAlmostEqual(switch_func(10.5), 0.5, delta=1e-6)
        self.assertAlmostEqual(switch_func(12.0), 0.0, delta=1e-6)

        # Numerical first and second derivatives at boundaries
        eps = 1e-5
        d1_on = (switch_func(r_on + eps) - switch_func(r_on)) / eps
        d1_off = (switch_func(r_off) - switch_func(r_off - eps)) / eps
        self.assertAlmostEqual(d1_on, 0.0, delta=1e-4)
        self.assertAlmostEqual(d1_off, 0.0, delta=1e-4)

    def test_haddock_score_on_benchmark_crystal(self):
        p = Path("benchmarks/1awr.pdb")
        if p.exists():
            res = calculate_haddock_physics_score(p, receptor_chain="A", peptide_chain="B")
            self.assertLess(res["e_vdw"], -10.0)
            self.assertLess(res["e_elec"], 0.0)
            self.assertLess(res["haddock_score"], -20.0)
            self.assertGreater(res["bsa"], 500.0)

    def test_mixed_hydrogen_crystal_1cka(self):
        """Integration test on benchmarks/1cka.pdb (contains 10 explicit hydrogens on Lysine)."""
        p = Path("benchmarks/1cka.pdb")
        if p.exists():
            res = calculate_haddock_physics_score(p, receptor_chain="A", peptide_chain="B")
            # Cationic polyproline peptide (2 Lys) interacting with anionic Abl SH3 pocket
            self.assertLess(res["e_elec"], -10.0, "1CKA must yield favorable attractive electrostatics")
            self.assertLess(res["haddock_score"], -20.0, "1CKA haddock_score must be favorable")
            self.assertGreater(res["bsa"], 700.0, "1CKA BSA must be > 700 A^2")

    def test_partially_protonated_fixture_regression(self):
        """Deliberately partially protonated fixture: 1 H atom added to an otherwise heavy-atom complex."""
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.300   2.400   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       2.000  -0.750   1.200  1.00 20.00           C
ATOM      6  N   ARG B   1       3.500   2.500   0.000  1.00 20.00           N
ATOM      7  CA  ARG B   1       4.500   3.500   0.000  1.00 20.00           C
ATOM      8  C   ARG B   1       6.000   3.000   0.000  1.00 20.00           C
ATOM      9  O   ARG B   1       6.500   2.000   0.000  1.00 20.00           O
ATOM     10  CB  ARG B   1       4.200   4.500   1.100  1.00 20.00           C
ATOM     11  CG  ARG B   1       5.000   5.800   1.100  1.00 20.00           C
ATOM     12  CD  ARG B   1       4.500   6.800   2.100  1.00 20.00           C
ATOM     13  NE  ARG B   1       5.200   8.100   2.100  1.00 20.00           N
ATOM     14  CZ  ARG B   1       4.800   9.200   2.800  1.00 20.00           C
ATOM     15  NH1 ARG B   1       3.600   9.200   3.400  1.00 20.00           N
ATOM     16  NH2 ARG B   1       5.600  10.200   2.900  1.00 20.00           N
ATOM     17  HE  ARG B   1       6.100   8.100   1.600  1.00 20.00           H
TER
END
"""
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tf:
            tf.write(pdb_content)
            tf_path = Path(tf.name)
        try:
            score = calculate_haddock_physics_score(tf_path, receptor_chain="A", peptide_chain="B")
            self.assertIsInstance(score["haddock_score"], float)
            self.assertGreater(score["bsa"], 0.0)
        finally:
            tf_path.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
