import unittest
from pathlib import Path
import tempfile
import gemmi
from esm_amp_screen.amp_analyzer import (
    calculate_net_charge,
    calculate_hydrophobic_moment,
    calculate_mean_hydrophobicity,
    calculate_boman_index,
    estimate_solution_affinity,
    analyze_cocomplex_contacts,
    analyze_complex_contacts
)
from esm_amp_screen.utils import clean_sequence

class TestAMPAnalyzer(unittest.TestCase):
    def test_net_charge(self):
        self.assertGreater(calculate_net_charge("RRRR"), 3.5)
        self.assertLess(calculate_net_charge("DDDD"), -3.5)
        self.assertAlmostEqual(calculate_net_charge("AAAA"), 0.0, delta=0.2)

    def test_hydrophobic_moment(self):
        mu_h = calculate_hydrophobic_moment("KWKKLLKKLLKLLKKLLK", angle_deg=100.0)
        self.assertGreater(mu_h, 0.2)

    def test_boman_index(self):
        boman = calculate_boman_index("GLFDIIKKIAESF")
        self.assertIsInstance(boman, float)

    def test_solution_affinity(self):
        res = estimate_solution_affinity(affinity_kcal_mol=-10.0, sequence="ACDEFGH", conformation="helix")
        self.assertIn("solution_delta_g", res)
        self.assertIn("solution_kd", res)
        self.assertGreater(res["solution_delta_g"], -10.0)

    def test_cocomplex_contacts(self):
        p = Path("benchmarks/1cka.pdb")
        if p.exists():
            contacts = analyze_cocomplex_contacts(p, receptor_chain="A", peptide_chain="B")
            self.assertGreater(contacts["hbonds"], 0)
            self.assertGreater(contacts["hydrophobic_contacts"], 0)

            # Intra-chain contacts must be strictly 0
            intra = analyze_cocomplex_contacts(p, receptor_chain="B", peptide_chain="B")
            self.assertEqual(intra["hbonds"], 0)
            self.assertEqual(intra["hydrophobic_contacts"], 0)
            self.assertEqual(intra["salt_bridges"], 0)

    def test_displaced_peptide_zero_contacts(self):
        """Translating the peptide by 50 A must result in zero contacts."""
        p = Path("benchmarks/1cka.pdb")
        if p.exists():
            st = gemmi.read_structure(str(p))
            for model in st:
                for chain in model:
                    if chain.name == "B":
                        for res in chain:
                            for atom in res:
                                atom.pos.x += 50.0
                                atom.pos.y += 50.0
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tf:
                tf_path = Path(tf.name)
            st.write_minimal_pdb(str(tf_path))
            try:
                c = analyze_cocomplex_contacts(tf_path, receptor_chain="A", peptide_chain="B")
                self.assertEqual(c["hbonds"], 0)
                self.assertEqual(c["hydrophobic_contacts"], 0)
                self.assertEqual(c["salt_bridges"], 0)
                self.assertEqual(c["clashes"], 0)
            finally:
                tf_path.unlink(missing_ok=True)

    def test_clean_sequence_unknown_residues(self):
        # Strict mode raises ValueError on X
        with self.assertRaises(ValueError):
            clean_sequence("ACDEFGHX", allow_unknown=False)
        # Permissive mode substitutes A with warning
        self.assertEqual(clean_sequence("ACDEFGHX", allow_unknown=True), "ACDEFGHA")

if __name__ == "__main__":
    unittest.main()
