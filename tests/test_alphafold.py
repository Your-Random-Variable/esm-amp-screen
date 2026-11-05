import unittest
import tempfile
from pathlib import Path
from esm_amp_screen.alphafold import (
    prepare_alphafold_multimer_inputs,
    score_alphafold_complex
)

class TestAlphaFoldModule(unittest.TestCase):
    def test_prepare_alphafold_multimer_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "test_af_inputs"
            target_seq = "ACDEFGHIKLMNPQRSTVWY"
            pep_seq = "KIFCGETCKFIPCITKVPGTCKCKSKVCYFN"
            
            res = prepare_alphafold_multimer_inputs(
                target_seq=target_seq,
                peptide_seq=pep_seq,
                job_name="unit_test_cpx",
                out_dir=out_dir
            )
            
            self.assertTrue(res["colabfold_fasta"].exists())
            self.assertTrue(res["af3_json"].exists())
            self.assertTrue(res["runner_script"].exists())
            
            fasta_txt = res["colabfold_fasta"].read_text()
            self.assertIn(f"{target_seq}:{pep_seq}", fasta_txt)
            
            json_txt = res["af3_json"].read_text()
            self.assertIn("proteinChain", json_txt)

    def test_score_alphafold_complex(self):
        cpx_path = Path("delivery_bama_docking/DesignB_BamA_docked_complex.pdb")
        if cpx_path.exists():
            res = score_alphafold_complex(
                complex_pdb=cpx_path,
                target_chain="A",
                peptide_chain="B"
            )
            self.assertIn("interface_plddt", res)
            self.assertIn("ml_delta_g", res)
            self.assertIn("decision", res)
            self.assertEqual(res["peptide_length"], 31)
            self.assertEqual(res["clashes"], 0)

if __name__ == "__main__":
    unittest.main()
