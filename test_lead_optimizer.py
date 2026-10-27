"""
Test suite for LeadOptimizer, interface relaxation, and in silico mutation scanning.
"""

from pathlib import Path
import gemmi
import numpy as np

from esm_amp_screen.lead_optimizer import (
    relax_complex_interface,
    mutate_peptide_residue,
    LeadOptimizer,
    AA_1TO3,
    HAS_OPENBABEL
)
from esm_amp_screen.physics_engine import calculate_haddock_physics_score

def test_mutate_peptide_residue():
    cpx_path = Path("benchmarks/1cka.pdb")
    assert cpx_path.exists(), "Benchmark 1CKA complex missing in benchmarks/"

    st = gemmi.read_structure(str(cpx_path))
    # Original residue 4 is ALA in chain B (index 3)
    orig_res = st[0]["B"][3]
    assert orig_res.name == "ALA"
    ca_orig = orig_res.find_atom("CA", "*").pos

    # Mutate to PHE (F)
    mutate_peptide_residue(st, "B", 3, "F")
    mut_res = st[0]["B"][3]
    assert mut_res.name == "PHE"
    ca_mut = mut_res.find_atom("CA", "*").pos

    # Verify backbone CA coordinate was preserved exactly
    dist = np.sqrt((ca_orig.x - ca_mut.x)**2 + (ca_orig.y - ca_mut.y)**2 + (ca_orig.z - ca_mut.z)**2)
    assert dist < 1e-4, f"Backbone CA moved during mutation (dist={dist})"

    # Verify all expected phenylalanine sidechain atoms are present
    atom_names = {a.name for a in mut_res}
    expected = {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    assert expected.issubset(atom_names), f"Missing PHE atoms: {expected - atom_names}"
    print("✓ test_mutate_peptide_residue passed.")

def test_interface_relaxation():
    cpx_path = Path("benchmarks/1cka.pdb")
    out_rel = Path("benchmarks/test_lead_opt_rel.pdb")
    out_rel.unlink(missing_ok=True)

    if not HAS_OPENBABEL:
        # Explicitly verify the graceful fallback copy behavior
        relax_complex_interface(cpx_path, out_rel, receptor_chain="A", peptide_chain="B", steps=15)
        assert out_rel.exists() and out_rel.stat().st_size > 0
        out_rel.unlink(missing_ok=True)
        print("⚠ OpenBabel Python bindings not available; tested copy-through fallback (live minimization skipped).")
        return

    # Run force-field relaxation
    relax_complex_interface(cpx_path, out_rel, receptor_chain="A", peptide_chain="B", steps=25)
    assert out_rel.exists() and out_rel.stat().st_size > 0

    # Verify that peptide coordinates actually shifted during force-field minimization
    st_orig = gemmi.read_structure(str(cpx_path))
    st_rel = gemmi.read_structure(str(out_rel))

    orig_pep = np.array([[a.pos.x, a.pos.y, a.pos.z] for m in st_orig for c in m if c.name == "B" for r in c for a in r])
    rel_pep = np.array([[a.pos.x, a.pos.y, a.pos.z] for m in st_rel for c in m if c.name == "B" for r in c for a in r])
    assert len(orig_pep) == len(rel_pep), "Peptide atom counts must match"
    pep_shifts = np.linalg.norm(orig_pep - rel_pep, axis=1)
    max_shift = float(np.max(pep_shifts))
    assert max_shift > 1e-3, f"Force-field relaxation did not move any peptide coordinates (max_shift={max_shift:.6f} A)"

    # Verify that receptor coordinates remained frozen
    orig_rec = np.array([[a.pos.x, a.pos.y, a.pos.z] for m in st_orig for c in m if c.name == "A" for r in c for a in r])
    rel_rec = np.array([[a.pos.x, a.pos.y, a.pos.z] for m in st_rel for c in m if c.name == "A" for r in c for a in r])
    rec_shift = float(np.max(np.linalg.norm(orig_rec - rel_rec, axis=1)))
    assert rec_shift < 1e-4, f"Receptor atoms moved during relaxation (max_rec_shift={rec_shift:.6f} A)"

    # Verify structure is written and can be scored
    score_rel = calculate_haddock_physics_score(out_rel, receptor_chain="A", peptide_chain="B")
    assert score_rel["haddock_score"] < 0.0, "Relaxed structure must have negative binding score"
    print(f"✓ test_interface_relaxation passed (max peptide shift: {max_shift:.3f} Å, receptor frozen, HADDOCK: {score_rel['haddock_score']:.1f}).")
    out_rel.unlink(missing_ok=True)

def test_lead_optimizer_execution():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        cpx_path = Path("benchmarks/1cka.pdb")
        out_dir = Path(tmp_dir) / "test_lead_opt"
        
        optimizer = LeadOptimizer(
            complex_pdb=cpx_path,
            receptor_chain="A",
            peptide_chain="B",
            out_dir=out_dir
        )

        # Test Alanine scan
        ala_res = optimizer.run_alanine_scan(threads=4)
        assert len(ala_res) == 9, f"Expected 9 residues in alanine scan, got {len(ala_res)}"
        
        # Verify hotspot detection on core interface Leucine (Leu5)
        l5_scan = [r for r in ala_res if r["position"] == 5][0]
        assert l5_scan["delta_haddock"] > 3.0, f"Leu5 was not identified as critical hotspot: {l5_scan['delta_haddock']}"

        # Test Lead Optimization on a specific position
        variants = optimizer.run_lead_optimization(
            candidate_positions=[4],
            candidate_amino_acids=["W", "Y", "F"],
            top_n=3,
            threads=4
        )
        assert len(variants) > 0, "No variants returned from lead optimization"
        top_var = variants[0]
        assert "mutation" in top_var
        assert "delta_delta_g" in top_var
        assert "mechanism" in top_var
        print(f"✓ test_lead_optimizer_execution passed (Top variant: {top_var['mutation']} ΔΔG={top_var['delta_delta_g']:.2f} kcal/mol).")

if __name__ == "__main__":
    test_mutate_peptide_residue()
    test_interface_relaxation()
    test_lead_optimizer_execution()
    print("\nALL LEAD OPTIMIZATION UNIT TESTS PASSED SUCCESSFULLY!")
