#!/usr/bin/env python3
"""
ESM-AMP-Screen Benchmark: Bound Co-Crystal Pose Rescoring & Sanity Check.
Evaluates the physics engine, solution thermodynamics, and contact analysis
on bound co-crystal poses across 6 standardized reference complexes from PepBDB / RCSB PDB:
1. 1AWR: Cyclophilin A + HAGPIA (6 aa)
2. 1CKA: Abl SH3 Domain + PPPALPPKK (9 aa)
3. 1DKX: DnaK Chaperone + NRLLLTG (7 aa)
4. 1ER8: Endothiapepsin + HPFHLLVY (8 aa)
5. 1SSH: c-Src SH3 Domain + GPPPAMPARPT (11 aa)
6. 1YCR: MDM2 Oncoprotein + ETFSDLWKLLPEN (13 aa)

Note: This benchmark verifies physical force-field rescoring and rank-order correlation
on known bound co-crystal geometries. It is a physical consistency sanity check, not a
prospective blind docking screen or calibrated absolute Kd predictor.
"""

import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import gemmi

from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import (
    analyze_cocomplex_contacts,
    estimate_solution_affinity,
    calculate_net_charge,
    calculate_hydrophobic_moment,
    calculate_boman_index
)
from esm_amp_screen.utils import AA_3TO1, logger

# Curated literature experimental reference data
# References:
# 1AWR: Zhao et al., Structure 5:139-146 (1997); CypA-substrate peptide Kd ~ 5.5 µM (ΔG = -7.18 kcal/mol)
# 1CKA: Musacchio et al., Science 263:971-974 (1994); Abl SH3 polyproline peptide Kd ~ 1.5 µM (ΔG = -7.95 kcal/mol)
# 1DKX: Zhu et al., Science 272:1606-1614 (1996); DnaK substrate peptide Kd ~ 0.4 µM (ΔG = -8.73 kcal/mol)
# 1ER8: Bailey et al., Protein Sci 2:1155-1162 (1993); Endothiapepsin transition-state inhibitor Kd ~ 12 nM (ΔG = -10.80 kcal/mol)
# 1SSH: Feng et al., Science 266:1241-1247 (1994); c-Src SH3 polyproline peptide Kd ~ 8.0 µM (ΔG = -6.95 kcal/mol)
# 1YCR: Kussie et al., Science 274:948-953 (1996); MDM2-p53 peptide Kd ~ 600 nM (ΔG = -8.49 kcal/mol)
LITERATURE_BENCHMARKS = {
    "1AWR": {
        "target": "Cyclophilin A (CypA)",
        "peptide_name": "HAGPIA substrate",
        "exp_kd_molar": 5.5e-6,
        "exp_kd_str": "5.5 µM",
        "exp_delta_g": -7.18
    },
    "1CKA": {
        "target": "Abl SH3 Domain",
        "peptide_name": "PPPALPPKK (Abl-p41)",
        "exp_kd_molar": 1.5e-6,
        "exp_kd_str": "1.5 µM",
        "exp_delta_g": -7.95
    },
    "1DKX": {
        "target": "DnaK Chaperone SBD",
        "peptide_name": "NRLLLTG (DnaK model)",
        "exp_kd_molar": 4.0e-7,
        "exp_kd_str": "0.4 µM",
        "exp_delta_g": -8.73
    },
    "1ER8": {
        "target": "Endothiapepsin",
        "peptide_name": "HPFHLLVY (H-142 analogue)",
        "exp_kd_molar": 1.2e-8,
        "exp_kd_str": "12 nM",
        "exp_delta_g": -10.80
    },
    "1SSH": {
        "target": "c-Src SH3 Domain",
        "peptide_name": "GPPPAMPARPT (Src-p41)",
        "exp_kd_molar": 8.0e-6,
        "exp_kd_str": "8.0 µM",
        "exp_delta_g": -6.95
    },
    "1YCR": {
        "target": "MDM2 Oncoprotein",
        "peptide_name": "ETFSDLWKLLPEN (p53 wt)",
        "exp_kd_molar": 6.0e-7,
        "exp_kd_str": "600 nM",
        "exp_delta_g": -8.49
    }
}


def run_all_benchmarks(benchmarks_dir: Path = Path("benchmarks"), out_csv: Path = Path("benchmarks/benchmark_results.csv")):
    """Evaluates all co-crystal structures in benchmarks_dir."""
    benchmarks_dir = Path(benchmarks_dir).resolve()
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    pdb_files = sorted(benchmarks_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {benchmarks_dir}")

    results = []

    print("\n" + "="*115)
    print("      ESM-AMP-SCREEN: STANDARDIZED CO-CRYSTAL BENCHMARK VALIDATION SUITE")
    print("="*115)
    print(f"Scanning benchmark directory: {benchmarks_dir} ({len(pdb_files)} complexes)\n")

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()
        meta = LITERATURE_BENCHMARKS.get(pdb_id, {
            "target": "Unknown Target",
            "peptide_name": pdb_id,
            "exp_kd_molar": None,
            "exp_kd_str": "N/A",
            "exp_delta_g": None
        })

        # 1. Read structure & extract peptide sequence from Chain B
        st = gemmi.read_structure(str(pdb_path))
        seq = "".join(
            AA_3TO1.get(res.name.upper(), "X")
            for m in st for c in m if c.name == "B"
            for res in c if res.name.upper() in AA_3TO1
        )

        # 2. Compute explicit physics terms
        phys = calculate_haddock_physics_score(pdb_path, receptor_chain="A", peptide_chain="B")
        e_vdw = phys["e_vdw"]
        e_elec = phys["e_elec"]
        e_desolv = phys["e_desolv"]
        haddock_score = phys["haddock_score"]
        bsa = phys["bsa"]

        # 3. Strictly cross-chain contact analysis (receptor chain A vs peptide chain B)
        contacts = analyze_cocomplex_contacts(pdb_path, receptor_chain="A", peptide_chain="B")
        hbonds = contacts["hbonds"]
        salt_bridges = contacts["salt_bridges"]
        hydrophobic = contacts["hydrophobic_contacts"]
        clashes = contacts["clashes"]

        # 4. Peptide biophysical metrics
        z = calculate_net_charge(seq) if seq else 0.0
        mu_h = calculate_hydrophobic_moment(seq) if seq else 0.0
        boman = calculate_boman_index(seq) if seq else 0.0

        # 5. Solution thermodynamics
        # Physics binding potential scaled into kcal/mol (0.28 scale factor converts composite HADDOCK into standard grid potential)
        grid_potential = haddock_score * 0.28
        sol_aff = estimate_solution_affinity(grid_potential, sequence=seq, conformation="auto")
        pred_delta_g = sol_aff["solution_delta_g"]
        pred_kd = sol_aff["solution_kd"]
        entropy_pen = sol_aff["entropy_penalty"]

        row = {
            "pdb_id": pdb_id,
            "target": meta["target"],
            "peptide_name": meta["peptide_name"],
            "sequence": seq,
            "length": len(seq),
            "net_charge_z": z,
            "e_vdw": round(e_vdw, 1),
            "e_elec": round(e_elec, 1),
            "e_desolv": round(e_desolv, 1),
            "haddock_score": round(haddock_score, 1),
            "bsa_angstrom2": round(bsa, 1),
            "hbonds": hbonds,
            "salt_bridges": salt_bridges,
            "hydrophobic_contacts": hydrophobic,
            "clashes": clashes,
            "entropy_penalty": round(entropy_pen, 2),
            "pred_delta_g": round(pred_delta_g, 2),
            "pred_kd": pred_kd,
            "exp_delta_g": meta["exp_delta_g"],
            "exp_kd": meta["exp_kd_str"]
        }
        results.append(row)

    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Print summary table
    header = f"{'PDB':<6} {'Target Receptor':<22} {'Seq (L)':<16} {'Evdw':<7} {'Eelec':<7} {'HADDOCK':<9} {'BSA(Å²)':<9} {'Pred ΔG':<9} {'Exp ΔG':<8} {'Pred Kd':<10} {'Exp Kd':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        seq_str = f"{r['sequence']} ({r['length']})"
        exp_dg_str = f"{r['exp_delta_g']:.2f}" if r['exp_delta_g'] is not None else "N/A"
        print(
            f"{r['pdb_id']:<6} {r['target'][:20]:<22} {seq_str:<16} "
            f"{r['e_vdw']:<7.1f} {r['e_elec']:<7.1f} {r['haddock_score']:<9.1f} "
            f"{r['bsa_angstrom2']:<9.0f} {r['pred_delta_g']:<9.2f} {exp_dg_str:<8} "
            f"{r['pred_kd']:<10} {r['exp_kd']:<8}"
        )
    print("-" * len(header))

    # Compute statistics if experimental data available
    valid = [r for r in results if r["exp_delta_g"] is not None]
    if len(valid) >= 3:
        pred_arr = np.array([r["pred_delta_g"] for r in valid])
        exp_arr = np.array([r["exp_delta_g"] for r in valid])
        r_val, p_val = pearsonr(pred_arr, exp_arr)
        rho_val, _ = spearmanr(pred_arr, exp_arr)
        rmse = float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))

        print("\nStatistical Validation vs Literature Experimental Affinities:")
        print(f"  • Pearson Correlation (r):  {r_val:.3f} (p = {p_val:.4f})")
        print(f"  • Spearman Rank Order (ρ):  {rho_val:.3f}")
        print(f"  • Root Mean Square Error:   {rmse:.2f} kcal/mol")
        print(f"  • Number of Co-Crystals:    {len(valid)}")
    print("="*115)
    print(f"\n[OK] Validation results saved to: {out_csv}\n")
    return df


if __name__ == "__main__":
    b_dir = Path(__file__).resolve().parent / "benchmarks"
    o_csv = b_dir / "benchmark_results.csv"
    run_all_benchmarks(b_dir, o_csv)
