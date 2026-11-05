import unittest
import tempfile
from pathlib import Path

from esm_amp_screen.structure_builder import (
    build_biophysical_peptide,
    fold_peptide,
    predict_secondary_structure_propensity,
    SECONDARY_STRUCTURE_DIHEDRALS
)

class TestStructureBuilder(unittest.TestCase):
    def test_predict_propensity(self):
        # Polyproline
        self.assertEqual(predict_secondary_structure_propensity("PPPPP"), "ppii")
        # Standard AMP amphipathic
        self.assertEqual(predict_secondary_structure_propensity("ALALALAKKK"), "helix")
        # Beta sheet dominant
        self.assertEqual(predict_secondary_structure_propensity("VVITFYWC"), "sheet")

    def test_build_biophysical_peptide(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_pdb = Path(tmp_dir) / "helix_pep.pdb"
            seq = "KIFCGETCKFIPCITKVPGTCKCKSKVCYFN"
            res_path = build_biophysical_peptide(seq, out_pdb, conformation="helix")
            self.assertTrue(res_path.exists())
            self.assertGreater(res_path.stat().st_size, 500)

            # Check fold_peptide auto
            out_auto = Path(tmp_dir) / "auto_pep.pdb"
            p, plddt = fold_peptide(seq, "test_amp", out_auto, conformation="auto")
            self.assertTrue(p.exists())
            self.assertGreater(plddt, 70.0)

if __name__ == "__main__":
    unittest.main()
