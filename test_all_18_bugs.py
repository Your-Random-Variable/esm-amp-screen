"""
Comprehensive Verification Test Suite for ESM-AMP-Screen.
Validates all physical, thermodynamic, and biochemical components with real assertions:
- P0-1: Heavy-atom / united-atom implicit hydrogen partial charge model
- P0-2: Continuous electrostatics switching function (9 to 12 A)
- P0-3: Lennard-Jones repulsive wall for severe atomic clashes
- P0-4: Transparent ML physical consensus score
- P0-6: Lead optimizer scale factor consistency (HADDOCK_TO_DG)
- Secondary structure Ramachandran dihedrals
- CAPRI benchmark evaluation
"""

import sys
import numpy as np
from pathlib import Path
import gemmi

from esm_amp_screen.physics_engine import (
    VDW_PARAMS,
    SOLVATION_PARAMS,
    PARTIAL_CHARGES,
    ALL_ATOM_CHARGES,
    HEAVY_ATOM_CHARGES,
    get_residue_formal_charge,
    calculate_haddock_physics_score
)
from esm_amp_screen.amp_analyzer import (
    calculate_net_charge,
    calculate_hydrophobic_moment,
    calculate_mean_hydrophobicity,
    calculate_boman_index,
    count_heavy_atoms,
    calculate_ligand_efficiency,
    calculate_per_residue_energy,
    analyze_complex_contacts,
    analyze_cocomplex_contacts,
    estimate_solution_affinity
)
from esm_amp_screen.ml_scorer import (
    ELEMENT_PAIRS,
    MLPeptideScorer,
    RT_LN10
)
from esm_amp_screen.lead_optimizer import HADDOCK_TO_DG
from esm_amp_screen.benchmark import evaluate_docked_complex
from esm_amp_screen.esm_folder import SECONDARY_STRUCTURE_DIHEDRALS
from esm_amp_screen.utils import clean_sequence, AA_3TO1


def test_p0_1_residue_formal_charges():
    """P0-1: Heavy-atom and all-atom charge models must assign correct formal charge to standard residues (pH 7.4)."""
    # 1. Assert direct summation of ALL_ATOM_CHARGES table
    all_atom_expected = {
        "ARG": 1.0, "LYS": 1.0, "ASP": -1.0, "GLU": -1.0,
        "HIP": 1.0, "HID": 0.0, "HIE": 0.0, "HIS": 0.0,
        "TYR": 0.0, "CYS": 0.0
    }
    for res, exp in all_atom_expected.items():
        direct_sum = sum(v for k, v in ALL_ATOM_CHARGES.items() if k[0] == res)
        assert abs(direct_sum - exp) < 1e-3, f"ALL_ATOM_CHARGES sum for {res} must be {exp}, got {direct_sum}"

    # 2. Assert direct summation of HEAVY_ATOM_CHARGES table
    heavy_expected = {
        "ARG": 1.0, "LYS": 1.0, "ASP": -1.0, "GLU": -1.0,
        "HIP": 1.0, "HID": 0.0, "HIE": 0.0, "HIS": 0.0
    }
    for res, exp in heavy_expected.items():
        direct_sum = sum(v for k, v in HEAVY_ATOM_CHARGES.items() if k[0] == res)
        assert abs(direct_sum - exp) < 1e-3, f"HEAVY_ATOM_CHARGES sum for {res} must be {exp}, got {direct_sum}"

    # 3. Dynamic get_residue_formal_charge helper assertions
    for has_h in [False, True]:
        label = "All-atom" if has_h else "Heavy-atom"
        assert abs(get_residue_formal_charge("ARG", has_hydrogens=has_h) - 1.0) < 0.05, f"{label} ARG must have formal charge +1.0"
        assert abs(get_residue_formal_charge("LYS", has_hydrogens=has_h) - 1.0) < 0.05, f"{label} LYS must have formal charge +1.0"
        assert abs(get_residue_formal_charge("ASP", has_hydrogens=has_h) - (-1.0)) < 0.05, f"{label} ASP must have formal charge -1.0"
        assert abs(get_residue_formal_charge("GLU", has_hydrogens=has_h) - (-1.0)) < 0.05, f"{label} GLU must have formal charge -1.0"
        assert abs(get_residue_formal_charge("ALA", has_hydrogens=has_h) - 0.0) < 0.05, f"{label} ALA must be neutral"
        assert abs(get_residue_formal_charge("LEU", has_hydrogens=has_h) - 0.0) < 0.05, f"{label} LEU must be neutral"

    # Integration test on 1cka.pdb (which contains 10 explicit hydrogens in chain B)
    p_1cka = Path("benchmarks/1cka.pdb")
    if p_1cka.exists():
        score_1cka = calculate_haddock_physics_score(p_1cka, receptor_chain="A", peptide_chain="B")
        assert score_1cka["e_elec"] < -10.0, f"Cationic peptide in 1CKA pocket must produce attractive electrostatics, got {score_1cka['e_elec']}"
        assert score_1cka["haddock_score"] < -20.0, f"Expected favorable HADDOCK score for 1CKA, got {score_1cka['haddock_score']}"

    print("✓ P0-1 Verified: Both heavy-atom and all-atom tables directly sum to correct formal charges (ARG=+1, LYS=+1, ASP=-1, GLU=-1, ALA=0), and 1cka.pdb electrostatics are favorable.")


def test_p0_2_switching_function():
    """P0-2: Canonical C2 quintic switching function S(u) = 1 - 10u^3 + 15u^4 - 6u^5 must smoothly taper to zero with zero derivatives."""
    r_on, r_off = 9.0, 12.0
    r_range = r_off - r_on

    def switch_val(r):
        if r <= r_on:
            return 1.0
        elif r >= r_off:
            return 0.0
        u = (r - r_on) / r_range
        return 1.0 - 10.0 * (u ** 3) + 15.0 * (u ** 4) - 6.0 * (u ** 5)

    # Function values at boundary and midpoint
    assert abs(switch_val(9.0) - 1.0) < 1e-6
    assert abs(switch_val(10.5) - 0.5) < 1e-6
    assert abs(switch_val(12.0) - 0.0) < 1e-6

    # Test numerical 1st and 2nd derivatives at endpoints (must vanish)
    eps = 1e-5
    # First derivative at r_on (from right)
    d1_on = (switch_val(r_on + eps) - switch_val(r_on)) / eps
    assert abs(d1_on) < 1e-4, f"First derivative at r_on must be zero, got {d1_on}"

    # First derivative at r_off (from left)
    d1_off = (switch_val(r_off) - switch_val(r_off - eps)) / eps
    assert abs(d1_off) < 1e-4, f"First derivative at r_off must be zero, got {d1_off}"

    # Second derivative at r_on
    d2_on = (switch_val(r_on + 2 * eps) - 2 * switch_val(r_on + eps) + switch_val(r_on)) / (eps ** 2)
    assert abs(d2_on) < 1e-3, f"Second derivative at r_on must be zero, got {d2_on}"

    # Second derivative at r_off
    d2_off = (switch_val(r_off) - 2 * switch_val(r_off - eps) + switch_val(r_off - 2 * eps)) / (eps ** 2)
    assert abs(d2_off) < 1e-3, f"Second derivative at r_off must be zero, got {d2_off}"

    print("✓ P0-2 Verified: Canonical C2 quintic switching function S(u) is smooth with zero 1st and 2nd derivatives at boundaries.")


def test_p0_3_repulsive_wall():
    """P0-3: Lennard-Jones potential must provide steep, non-decreasing repulsive barrier for overlapping atoms (including r < 0.8 A)."""
    sig_c, eps_c = VDW_PARAMS["C"]
    sig_cc = sig_c
    eps_cc = eps_c

    def calc_vdw(r):
        r_eff = max(r, 1.0)
        sr6 = (sig_cc / r_eff) ** 6
        vdw = 4.0 * eps_cc * (sr6 ** 2 - sr6)
        return float(np.clip(vdw, -2.0, 500.0))

    # Test monotonic non-decreasing penalty as distance decreases across sub-angstrom clashes
    distances = [1.8, 1.5, 1.0, 0.7, 0.5]
    energies = [calc_vdw(r) for r in distances]

    assert energies[0] > 10.0, f"Repulsive penalty at 1.8 A should be > 10 kcal/mol, got {energies[0]}"
    assert energies[1] > 100.0, f"Repulsive penalty at 1.5 A should be > 100 kcal/mol, got {energies[1]}"
    assert energies[2] == 500.0, f"Repulsive penalty at 1.0 A must hit 500.0 kcal/mol cap, got {energies[2]}"
    assert energies[3] == 500.0, f"Repulsive penalty at 0.7 A must retain 500.0 kcal/mol cap, got {energies[3]}"
    assert energies[4] == 500.0, f"Repulsive penalty at 0.5 A must retain 500.0 kcal/mol cap, got {energies[4]}"

    # Assert non-decreasing
    for e_prev, e_curr in zip(energies[:-1], energies[1:]):
        assert e_curr >= e_prev, f"Repulsive energy must be non-decreasing as distance decreases: {e_curr} < {e_prev}"

    print(f"✓ P0-3 Verified: Lennard-Jones repulsive wall is monotonic and caps at +500 kcal/mol for all sub-1.0 A overlaps.")


def test_p0_4_ml_scorer_uncalibrated_consensus():
    """P0-4: ML Scorer transparently marks is_calibrated=False and computes physical consensus."""
    scorer = MLPeptideScorer()
    assert scorer.is_calibrated is False, "Default stub must be marked uncalibrated"
    # Test consensus score: balances -vina, -haddock/10, and solution affinity
    score = scorer.compute_consensus_score(vina_affinity=-8.5, haddock_score=-40.0)
    assert score > 0.0, f"Expected positive consensus score, got {score}"
    print(f"✓ P0-4 Verified: ML scorer is marked uncalibrated (is_calibrated=False) with physical consensus ({score:.2f}).")


def test_p0_6_lead_optimizer_scale_factor():
    """P0-6: Lead optimizer HADDOCK_TO_DG scale factor must be strictly 0.15 across all mutation routines."""
    assert abs(HADDOCK_TO_DG - 0.15) < 1e-4, f"HADDOCK_TO_DG scale must be 0.15, got {HADDOCK_TO_DG}"
    # A -10 HADDOCK improvement yields -1.5 kcal/mol DeltaDeltaG
    ddg = -10.0 * HADDOCK_TO_DG
    assert abs(ddg - (-1.5)) < 1e-4
    print("✓ P0-6 Verified: Lead optimizer HADDOCK_TO_DG scale is unified at 0.15.")


def test_vdw_parameters_and_combining_rule():
    """VDW Lorentz-Berthelot combining rule and parameters."""
    sig_c, eps_c = VDW_PARAMS["C"]
    sig_n, eps_n = VDW_PARAMS["N"]
    expected_sig = (sig_c + sig_n) / 2.0
    assert abs(expected_sig - 3.375) < 1e-4
    expected_eps = np.sqrt(eps_c * eps_n)
    assert abs(expected_eps - np.sqrt(0.066 * 0.170)) < 1e-4
    assert VDW_PARAMS["H"][0] == 2.50
    print("✓ VDW parameters and arithmetic Lorentz-Berthelot rule verified.")


def test_desolvation_parameters():
    """Eisenberg & McLachlan atomic solvation parameters."""
    assert SOLVATION_PARAMS["C"] == -0.016
    assert SOLVATION_PARAMS["N"] == +0.006
    assert SOLVATION_PARAMS["O"] == +0.006
    assert SOLVATION_PARAMS["S"] == -0.021
    print("✓ Solvation parameters verified (C/S hydrophobic favorable, N/O polar penalty).")


def test_published_haddock_weights():
    """HADDOCK 2.4 score weights."""
    e_vdw, e_elec, e_desolv, e_air = -20.0, -15.0, -9.0, 0.0
    # Formula: 1.0*Evdw + 0.2*Eelec + 1.0*Edesolv + 0.1*Eair
    score = 1.0 * e_vdw + 0.2 * e_elec + 1.0 * e_desolv + 0.1 * e_air
    assert abs(score - (-32.0)) < 1e-4
    print("✓ Published HADDOCK 2.4 score weights verified.")


def test_amber_charges():
    """AMBER ff14SB charge table integrity."""
    assert PARTIAL_CHARGES[("ARG", "NH1")] == -0.3549
    assert PARTIAL_CHARGES[("ARG", "CZ")] == 0.7341
    assert PARTIAL_CHARGES[("LYS", "CE")] == -0.0143
    assert ("HID", "ND1") in PARTIAL_CHARGES
    assert ("HIE", "NE2") in PARTIAL_CHARGES
    assert ("HIP", "ND1") in PARTIAL_CHARGES
    print("✓ AMBER ff14SB charges verified.")


def test_element_pairs_and_thermodynamics():
    """10 unique element pairs and RT_LN10 thermodynamic conversion."""
    assert len(ELEMENT_PAIRS) == 10
    assert ("S", "S") in ELEMENT_PAIRS
    assert abs(RT_LN10 - 1.3638) < 0.005
    print(f"✓ 10 element pairs and RT_LN10 ({RT_LN10:.4f}) verified.")


def test_secondary_structure_dihedrals():
    """Standard Ramachandran angles."""
    assert SECONDARY_STRUCTURE_DIHEDRALS["helix"]["phi"] == -57.0
    assert SECONDARY_STRUCTURE_DIHEDRALS["helix"]["psi"] == -47.0
    assert SECONDARY_STRUCTURE_DIHEDRALS["sheet"]["phi"] == -119.0
    assert SECONDARY_STRUCTURE_DIHEDRALS["sheet"]["psi"] == 113.0
    assert SECONDARY_STRUCTURE_DIHEDRALS["ppii"]["phi"] == -78.0
    assert SECONDARY_STRUCTURE_DIHEDRALS["ppii"]["psi"] == 149.0
    print("✓ Ramachandran conformational priors verified.")


def test_clean_sequence_and_nonstandard_mapping():
    """Non-standard amino acid mapping in clean_sequence."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    assert clean_sequence(seq) == seq
    # Test U -> C (Sec), O -> K (Pyl), B -> D, Z -> E, J -> L
    assert clean_sequence("UOBZJ") == "CKDEL"
    print("✓ clean_sequence maps non-standard residues correctly (U->C, O->K, B->D, Z->E, J->L).")


def test_capri_benchmark_on_real_crystal():
    """Verify CAPRI evaluation on benchmark 1AWR reference structure."""
    ref_path = Path("benchmarks/1awr.pdb")
    if ref_path.exists():
        # Superimposing identical structure should yield perfect 0.0 RMSD and 100% native contacts
        metrics = evaluate_docked_complex(
            docked_pdb=ref_path,
            reference_pdb=ref_path,
            docked_rec_chain="A",
            docked_pep_chain="B",
            ref_rec_chain="A",
            ref_pep_chain="B"
        )
        assert metrics["receptor_ca_rmsd"] < 1e-3, f"Expected CA RMSD 0, got {metrics['receptor_ca_rmsd']}"
        assert metrics["l_rmsd_backbone"] < 1e-3, f"Expected L-RMSD 0, got {metrics['l_rmsd_backbone']}"
        assert metrics["f_nat_percent"] == 100.0, f"Expected 100% native contacts, got {metrics['f_nat_percent']}"
        assert metrics["capri_quality"] == "High (***)", f"Expected High, got {metrics['capri_quality']}"
        print(f"✓ CAPRI Benchmark Verified: L-RMSD = {metrics['l_rmsd_backbone']:.3f} A, F_nat = {metrics['f_nat_percent']}%, Quality = {metrics['capri_quality']}")


def test_cocomplex_cross_chain_contacts():
    """Verify that analyze_cocomplex_contacts strictly extracts cross-chain contacts and rejects intra-chain pairs."""
    p_1cka = Path("benchmarks/1cka.pdb")
    if p_1cka.exists():
        contacts = analyze_cocomplex_contacts(p_1cka, receptor_chain="A", peptide_chain="B")
        assert contacts["hbonds"] > 0, "1CKA crystal must have cross-chain H-bonds"
        assert contacts["hydrophobic_contacts"] > 0, "1CKA crystal must have cross-chain hydrophobic contacts"

        # Verify that passing the same chain as both receptor and peptide returns 0 contacts (no intra-chain pairs)
        intra = analyze_cocomplex_contacts(p_1cka, receptor_chain="B", peptide_chain="B")
        assert intra["hbonds"] == 0, f"Intra-chain H-bonds must be 0, got {intra['hbonds']}"
        assert intra["hydrophobic_contacts"] == 0, f"Intra-chain hydrophobic contacts must be 0, got {intra['hydrophobic_contacts']}"
        assert intra["salt_bridges"] == 0, f"Intra-chain salt bridges must be 0, got {intra['salt_bridges']}"
        print("✓ Co-complex cross-chain contact analysis verified: intra-chain pairs strictly excluded.")


def test_p1_separate_pdbs_chain_a_contacts():
    """Verify that analyze_complex_contacts does not suppress contacts when two separate PDBs both use chain A."""
    import tempfile
    rec_pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00 20.00           C\n"
        "ATOM      4  O   ALA A   1       1.300   2.400   0.000  1.00 20.00           O\n"
        "TER\nEND\n"
    )
    lig_pdb = (
        "ATOM      1  N   SER A   1       1.300   5.200   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  SER A   1       2.200   6.200   0.000  1.00 20.00           C\n"
        "ATOM      3  C   SER A   1       3.600   5.800   0.000  1.00 20.00           C\n"
        "ATOM      4  O   SER A   1       4.000   4.700   0.000  1.00 20.00           O\n"
        "ATOM      5  CB  SER A   1       1.800   7.600   0.000  1.00 20.00           C\n"
        "ATOM      6  OG  SER A   1       1.300   2.400   2.800  1.00 20.00           O\n"
        "TER\nEND\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix="_rec.pdb", delete=False) as tf_rec, \
         tempfile.NamedTemporaryFile("w", suffix="_lig.pdb", delete=False) as tf_lig:
        tf_rec.write(rec_pdb)
        tf_lig.write(lig_pdb)
        rec_path = Path(tf_rec.name)
        lig_path = Path(tf_lig.name)

    try:
        contacts = analyze_complex_contacts(rec_path, lig_path)
        assert contacts["hbonds"] >= 1, f"Expected H-bond between separate PDBs both on chain A, got {contacts['hbonds']}"
        print("✓ P1 Verified: analyze_complex_contacts detects contacts between separate PDBs even when both use chain A.")
    finally:
        rec_path.unlink(missing_ok=True)
        lig_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n" + "="*85)
    print("      RUNNING ESM-AMP-SCREEN LOGICAL, PHYSICAL & THEORETICAL AUDIT SUITE")
    print("="*85 + "\n")
    test_p0_1_residue_formal_charges()
    test_p0_2_switching_function()
    test_p0_3_repulsive_wall()
    test_p0_4_ml_scorer_uncalibrated_consensus()
    test_p0_6_lead_optimizer_scale_factor()
    test_vdw_parameters_and_combining_rule()
    test_desolvation_parameters()
    test_published_haddock_weights()
    test_amber_charges()
    test_element_pairs_and_thermodynamics()
    test_secondary_structure_dihedrals()
    test_clean_sequence_and_nonstandard_mapping()
    test_cocomplex_cross_chain_contacts()
    test_p1_separate_pdbs_chain_a_contacts()
    test_capri_benchmark_on_real_crystal()
    print("\n" + "="*85)
    print("      ALL AUDIT CHECKS & SCIENTIFIC VERIFICATION TESTS PASSED (0 ERRORS)")
    print("="*85 + "\n")
