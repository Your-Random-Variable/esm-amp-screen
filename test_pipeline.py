"""
Automated Comprehensive Integration & Feature Test for ESM-AMP-Screen.
Validates:
1. Auto-Pocket geometric cavity detector
2. Disulfide bond and macrocycle locking
3. Multi-conformation ensemble generation
4. End-to-end 3-Stage Screening funnel (HTVS + Sidechain Refinement + HADDOCK/ML rescoring)
5. Standalone Interactive HTML 3D Report generation
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import shutil
import pytest
import gemmi

from esm_amp_screen.receptor_prep import detect_surface_cavities, calculate_grid_box
from esm_amp_screen.peptide_prep import detect_and_annotate_disulfides, prepare_peptide_pdbqt
from esm_amp_screen.esm_folder import batch_fold_peptides

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Robust tool detection (PATH or python prefix)
OBABEL = shutil.which("obabel") or (str(Path(sys.prefix) / "bin" / "obabel") if (Path(sys.prefix) / "bin" / "obabel").exists() else None)
VINA = shutil.which("vina") or (str(Path(sys.prefix) / "bin" / "vina") if (Path(sys.prefix) / "bin" / "vina").exists() else None)

HAS_OBABEL = OBABEL is not None
HAS_VINA = VINA is not None


def test_auto_pocket():
    print("\n--- Testing Enhancement 3: Auto-Pocket Detection ---")
    target_pdb = ROOT / "examples" / "receptors" / "target.pdb"
    pockets = detect_surface_cavities(target_pdb, max_pockets=3)
    assert len(pockets) >= 1, "Failed to detect surface cavities in target.pdb"
    p1 = pockets[0]
    print(f"[PASS] Pocket #1 detected: {p1['num_points']} points, center=({p1['center_x']}, {p1['center_y']}, {p1['center_z']}), {len(p1['contact_residues'])} residues.")
    
    # Test calculate_grid_box with auto_pocket
    grid = calculate_grid_box(target_pdb, auto_pocket=True)
    assert "center_x" in grid and "detected_pocket_residues" in grid
    print(f"[PASS] Grid box auto-pocket center: ({grid['center_x']}, {grid['center_y']}, {grid['center_z']})")


@pytest.mark.skipif(not HAS_OBABEL, reason="OpenBabel binary not available")
def test_disulfide_locking():
    print("\n--- Testing Enhancement 4: Disulfide & Macrocycle Locking ---")
    if not HAS_OBABEL:
        pytest.skip("OpenBabel binary not found")
    # Synthetic cyclic peptide with Cys1-Cys2 disulfide
    pdb_text = """ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  CYS A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   CYS A   1       2.000   1.400   0.000  1.00 20.00           C
ATOM      4  O   CYS A   1       1.300   2.400   0.000  1.00 20.00           O
ATOM      5  CB  CYS A   1       2.000  -0.750   1.200  1.00 20.00           C
ATOM      6  SG  CYS A   1       1.500  -2.500   1.200  1.00 20.00           S
ATOM      7  N   CYS A   2       3.300   1.400   0.000  1.00 20.00           N
ATOM      8  CA  CYS A   2       4.000   2.700   0.000  1.00 20.00           C
ATOM      9  C   CYS A   2       5.500   2.500   0.000  1.00 20.00           C
ATOM     10  O   CYS A   2       6.200   3.500   0.000  1.00 20.00           O
ATOM     11  CB  CYS A   2       3.600   3.500   1.200  1.00 20.00           C
ATOM     12  SG  CYS A   2       1.500  -0.500   1.200  1.00 20.00           S
END
"""
    test_pdb = ROOT / "test_cyclic.pdb"
    test_annotated = ROOT / "test_cyclic.annotated.pdb"
    test_pdbqt = ROOT / "test_cyclic.pdbqt"
    try:
        test_pdb.write_text(pdb_text)
        disulfides, ann_path = detect_and_annotate_disulfides(test_pdb, test_annotated)
        assert len(disulfides) == 1, "Failed to perceive CYS1-CYS2 disulfide bond!"
        print(f"[PASS] Disulfide perceived: {disulfides[0]['cys1']}-{disulfides[0]['cys2']} (d={disulfides[0]['distance']} Å)")

        out_pdbqt, torsions = prepare_peptide_pdbqt(
            test_pdb,
            test_pdbqt,
            flexibility="sidechain",
            obabel_bin=OBABEL,
            lock_disulfides=True
        )
        assert out_pdbqt.exists(), "PDBQT generation failed for cyclic peptide"
        print(f"[PASS] Disulfide-locked PDBQT prepared successfully (active torsions: {torsions})")
    finally:
        for p in (test_pdb, test_annotated, test_pdbqt):
            if p.exists():
                p.unlink(missing_ok=True)


def test_ensemble_folding():
    print("\n--- Testing Enhancement 2: Multi-Conformation Ensemble Folding ---")
    test_fa = {"MiniAMP": "RRWWRF"}
    temp_dir = ROOT / "test_ensemble_pdbs"
    temp_dir.mkdir(exist_ok=True)
    try:
        folded = batch_fold_peptides(test_fa, temp_dir, conformation="ensemble")
        # Should have 3 conformers: helix, extended, ppii
        assert len(folded) == 3, f"Expected 3 ensemble conformers, got {len(folded)}"
        keys = list(folded.keys())
        print(f"[PASS] Generated ensemble conformers: {keys}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not (HAS_OBABEL and HAS_VINA), reason="OpenBabel or Vina binary not available")
def test_end_to_end_pipeline():
    print("\n--- Testing End-to-End Automated Pipeline (Auto-Pocket + 3 Stages + HTML Report) ---")
    if not (HAS_OBABEL and HAS_VINA):
        pytest.skip("OpenBabel or Vina binary not found")

    target_pdb = ROOT / "examples" / "receptors" / "target.pdb"
    peptides_fa = ROOT / "examples" / "peptides" / "benchmark_amps.fasta"
    out_dir = ROOT / "screening_results_test"

    cmd = [
        PYTHON, "-m", "esm_amp_screen.cli", "run",
        "--receptor", str(target_pdb),
        "--library", str(peptides_fa),
        "--output", str(out_dir),
        "--auto-pocket",
        "--flexibility", "rigid",
        "--max-workers", "2",
        "--exhaustiveness", "2",
        "--refine-top", "2",
        "--refine-exhaustiveness", "2",
        "--top", "3",
    ]
    if OBABEL:
        cmd.extend(["--obabel-bin", OBABEL])

    print("Running command:", " ".join(cmd))
    res = subprocess.run(cmd)
    assert res.returncode == 0, f"Screening pipeline failed with exit code {res.returncode}"

    csv_report = out_dir / "screening_leaderboard.csv"
    html_report = out_dir / "screening_report.html"
    pml_script = out_dir / "view_hits.pml"
    complex_dir = out_dir / "top_complexes"

    assert csv_report.exists(), "Missing screening_leaderboard.csv"
    assert html_report.exists() and html_report.stat().st_size > 50000, "Missing or empty screening_report.html"
    assert pml_script.exists(), "Missing view_hits.pml"
    assert complex_dir.exists() and len(list(complex_dir.glob("*.pdb"))) > 0, "Missing complex PDBs"

    df = pd.read_csv(csv_report)
    print(f"[PASS] Screening leaderboard columns: {df.columns.tolist()[:8]}...")
    assert "haddock_score" in df.columns, "Missing haddock_score column"
    assert "ml_pkd" in df.columns, "Missing ml_pkd column"
    assert "consensus_score" in df.columns, "Missing consensus_score column"
    print(f"[PASS] Standalone HTML Report verified ({html_report.stat().st_size // 1024} KB)")


def main():
    test_auto_pocket()
    test_disulfide_locking()
    test_ensemble_folding()
    test_end_to_end_pipeline()
    print("\n" + "="*80)
    print(">>> ALL 4 ADVANCED ENHANCEMENTS AND PIPELINE PHASES VERIFIED 100%! <<<")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
