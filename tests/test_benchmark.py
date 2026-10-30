import unittest
import tempfile
from pathlib import Path
from esm_amp_screen.benchmark import evaluate_docked_complex
from run_benchmarks import run_all_benchmarks

class TestBenchmark(unittest.TestCase):
    def test_capri_evaluation_benchmark(self):
        pdb_path = Path("benchmarks/1awr.pdb")
        if pdb_path.exists():
            metrics = evaluate_docked_complex(
                docked_pdb=pdb_path,
                reference_pdb=pdb_path,
                docked_rec_chain="A",
                docked_pep_chain="B",
                ref_rec_chain="A",
                ref_pep_chain="B"
            )
            self.assertLess(metrics["receptor_ca_rmsd"], 1e-3)
            self.assertLess(metrics["l_rmsd_backbone"], 1e-3)
            self.assertEqual(metrics["f_nat_percent"], 100.0)
            self.assertEqual(metrics["capri_quality"], "High (***)")

    def test_run_all_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = Path(tmp_dir) / "benchmark_results.csv"
            df = run_all_benchmarks(benchmarks_dir=Path("benchmarks"), out_csv=out_csv)
            self.assertGreaterEqual(len(df), 6)
            self.assertEqual(len(df), len(list(Path("benchmarks").glob("*.pdb"))))
            self.assertTrue(out_csv.exists())
            self.assertIn("pdb_id", df.columns)
            self.assertIn("haddock_score", df.columns)

if __name__ == "__main__":
    unittest.main()
