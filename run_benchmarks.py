#!/usr/bin/env python3
"""
ESM-AMP-Screen Benchmark: Bound Co-Crystal Pose Rescoring & Sanity Check.
Evaluates the physics engine, solution thermodynamics, and contact analysis
on bound co-crystal poses across 6 standardized reference complexes from PepBDB / RCSB PDB:
1. 1AWR: Cyclophilin A + HAGPIA (6 aa)
2. 1CKA: c-Crk N-SH3 + PPPALPPKK (9 aa resolved-core proxy for 10-mer C3G)
3. 1DKX: DnaK Chaperone SBD (apo, 25°C) + NRLLLTG (7 aa)
4. 1ER8: Endothiapepsin + H-77 peptidomimetic (8 aa)
5. 1SSH: Yeast YFR024C-A SH3 + Sla1 peptide (11 aa)
6. 1YCR: MDM2 Oncoprotein + p53(17-29) (13 aa; 15-mer ITC proxy, 35°C)

Note: This benchmark verifies physical force-field rescoring and rank-order correlation
on known bound co-crystal geometries. It is an exploratory physical consistency sanity check
(N=4 descriptive pose-affinity comparisons including structural proxies), not a prospective
blind docking screen or calibrated absolute Kd predictor.
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
from esm_amp_screen.ml_scorer import MLPeptideScorer

# Curated literature experimental reference data and structural provenance:
# 1AWR: Vajdos et al., Protein Sci 6:2297-2307 (1997) [structure]; CypA active site + HIV-1 CA (HAGPIA, 6 aa; residues 87-92).
#       Solution affinity context from HIV-1 capsid binding studies (Kd ~ 5.5 - 16 µM, e.g. Yoo et al., JMB 1997; Gamble et al., Cell 1996; Vajdos et al. 1997), ΔG = -7.18 kcal/mol.
# 1CKA: Wu et al., Structure 3:215-226 (1995); Knudsen et al., EMBO J 13:827 (1994); Murine c-Crk N-SH3 + C3G peptide.
#       Crystal pose resolves 9-mer PPPALPPKK (C-terminal Arg disordered), evaluated as a resolved-core proxy for measured 10-mer PPPALPPKKR, Kd ~ 2.0 µM (ΔG = -7.77 kcal/mol).
# 1DKX: Zhu et al., Science 272:1606-1614 (1996); Burkholder et al., PNAS 93:10632 (1996); E. coli DnaK SBD (residues 389-607) in the apo (nucleotide-free) state
#       + λ O peptide (NRLLLTG, 7 aa), determined by fluorescence anisotropy at 25°C, Kd ~ 0.4 µM (ΔG = -8.73 kcal/mol). Affinity shifts significantly in ATP-bound state.
# 1YCR: Kussie et al., Science 274:948-953 (1996) [structure]; Dawson et al., Mol Cancer Res 2:20-30 (2004) [ITC affinity];
#       MDM2 + p53 core (ETFSDLWKLLPEN, 13 aa resolved core proxy for 15-mer SQETFSDLWKLLPEN); Kd ~ 600 nM determined by ITC at 35°C (308.15 K; ΔG° = -8.77 kcal/mol; -8.49 kcal/mol if 298.15 K conversion).
# 1SSH: SGX Structural Genomics Consortium (PDB ID: 1SSH, deposited 2004, released 2005); S. cerevisiae YFR024C-A SH3 + Sla1-derived peptide (11 resolved aa of 12-mer).
#       Retained as structural pose check (no peer-reviewed solution Kd; 8.0 µM was chicken c-Src 1PRL).
# 1ER8: Cooper et al., FEBS Lett 174:96-101 (1984); Bailey et al., Protein Sci 2:1155 (1993); Endothiapepsin + H-77 peptidomimetic
#       (D-His-Pro-Phe-His-Leu-[CH2-NH]-Leu-Val-Tyr, 8 aa) containing reduced isostere bond. (12 nM was canine renin IC50).
LITERATURE_BENCHMARKS = {
    "1AWR": {
        "target": "Cyclophilin A (CypA)",
        "peptide_name": "HIV-1 CA (HAGPIA)",
        "exp_kd_molar": 5.5e-6,
        "exp_kd_str": "5.5 µM",
        "exp_delta_g": -7.18,
        "citation": "Vajdos et al., Protein Sci. 6:2297 (1997)"
    },
    "1CKA": {
        "target": "c-Crk N-SH3 Domain",
        "peptide_name": "C3G (9-mer core proxy)",
        "exp_kd_molar": 2.0e-6,
        "exp_kd_str": "2.0 µM*",
        "exp_delta_g": -7.77,
        "citation": "Wu et al., Structure 3:215 (1995)"
    },
    "1DKX": {
        "target": "DnaK SBD (apo, 25°C)",
        "peptide_name": "λ O model (NRLLLTG)",
        "exp_kd_molar": 4.0e-7,
        "exp_kd_str": "0.4 µM",
        "exp_delta_g": -8.73,
        "citation": "Zhu et al., Science 272:1606 (1996)"
    },
    "1ER8": {
        "target": "Endothiapepsin",
        "peptide_name": "H-77 Peptidomimetic",
        "exp_kd_molar": None,
        "exp_kd_str": "Peptidomimetic",
        "exp_delta_g": None,
        "citation": "Bailey et al., Protein Sci. 2:1155 (1993)"
    },
    "1SSH": {
        "target": "Yeast YFR024C-A SH3",
        "peptide_name": "Sla1 (GPPPAMPARPT)",
        "exp_kd_molar": None,
        "exp_kd_str": "Unreported",
        "exp_delta_g": None,
        "citation": "SGX Consortium (2004)"
    },
    "1YCR": {
        "target": "MDM2 Oncoprotein",
        "peptide_name": "p53 (15-mer ITC proxy)",
        "exp_kd_molar": 6.0e-7,
        "exp_kd_str": "600 nM* (35°C)",
        "exp_delta_g": -8.49,
        "citation": "Kussie 1996 / Dawson 2004"
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

    ml_model_file = Path(__file__).resolve().parent / "data" / "trained_complex_ensemble.joblib"
    ml_scorer = MLPeptideScorer(model_path=ml_model_file) if ml_model_file.exists() else None

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

        # 6. Calibrated 34D ML Ensemble Regressor
        if ml_scorer and ml_scorer.is_calibrated:
            ml_res = ml_scorer.score_complex(
                pdb_path,
                hit_meta={
                    "pep_id": pdb_id,
                    "sequence": seq,
                    "length": len(seq),
                    "affinity": pred_delta_g,
                    "docking_energy": pred_delta_g,
                    "hbonds": hbonds,
                    "salt_bridges": salt_bridges,
                    "hydrophobic_contacts": hydrophobic,
                    "clashes": clashes,
                    "net_charge_z": z,
                    "hydrophobic_moment_muH": mu_h,
                    "boman_index": boman,
                }
            )
            ml_delta_g = ml_res["ml_delta_g"]
            ml_kd = ml_res["ml_kd"]
            ml_pkd = ml_res["ml_pkd"]
        else:
            ml_delta_g = None
            ml_kd = "N/A"
            ml_pkd = None

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
            "ml_delta_g": round(ml_delta_g, 2) if ml_delta_g is not None else None,
            "ml_kd": ml_kd,
            "ml_pkd": ml_pkd,
            "exp_delta_g": meta["exp_delta_g"],
            "exp_kd": meta["exp_kd_str"]
        }
        results.append(row)

    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Print summary table
    header = f"{'PDB':<6} {'Target Receptor':<20} {'Seq (L)':<15} {'HADDOCK':<9} {'BSA(Å²)':<8} {'Phys ΔG':<8} {'ML ΔG':<8} {'Exp ΔG':<8} {'Phys Kd':<10} {'ML Kd':<10} {'Exp Kd':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        seq_str = f"{r['sequence']} ({r['length']})"
        exp_dg_str = f"{r['exp_delta_g']:.2f}" if r['exp_delta_g'] is not None else "N/A"
        ml_dg_str = f"{r['ml_delta_g']:.2f}" if r['ml_delta_g'] is not None else "N/A"
        print(
            f"{r['pdb_id']:<6} {r['target'][:18]:<20} {seq_str:<15} "
            f"{r['haddock_score']:<9.1f} {r['bsa_angstrom2']:<8.0f} "
            f"{r['pred_delta_g']:<8.2f} {ml_dg_str:<8} {exp_dg_str:<8} "
            f"{r['pred_kd']:<10} {r['ml_kd']:<10} {r['exp_kd']:<8}"
        )
    print("-" * len(header))

    # Compute descriptive statistics on retrospective pose-affinity comparisons (including structural proxies)
    valid = [r for r in results if r["exp_delta_g"] is not None]
    if len(valid) >= 3:
        pred_arr = np.array([r["pred_delta_g"] for r in valid])
        exp_arr = np.array([r["exp_delta_g"] for r in valid])
        r_val, p_val = pearsonr(pred_arr, exp_arr)
        rho_val, _ = spearmanr(pred_arr, exp_arr)
        rmse = float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - exp_arr)))

        print("\nDescriptive Statistics for Four Retrospective Pose–Affinity Comparisons (Including Structural Proxies):")
        print("  • Note: n=4 exploratory sample with structural proxies (1CKA 9-mer vs 10-mer; 1YCR 13-mer vs 15-mer at 35°C).")
        print("    Descriptive metrics only; does not establish independent predictive validation.")
        print(f"  • Matched / Proxy Pairs:     {len(valid)} ({', '.join(r['pdb_id'] for r in valid)})")
        print(f"  • Physical Force-Field:      MAE = {mae:.2f} kcal/mol, RMSE = {rmse:.2f} kcal/mol, r = {r_val:.3f} (p = {p_val:.4f}), ρ = {rho_val:.3f}")

        if any(r["ml_delta_g"] is not None for r in valid):
            ml_arr = np.array([r["ml_delta_g"] for r in valid])
            ml_mae = float(np.mean(np.abs(ml_arr - exp_arr)))
            ml_rmse = float(np.sqrt(np.mean((ml_arr - exp_arr) ** 2)))
            print(f"  • 56D Trained ML Ensemble:   MAE = {ml_mae:.2f} kcal/mol, RMSE = {ml_rmse:.2f} kcal/mol")
    print("="*115)
    print(f"\n[OK] Validation results saved to: {out_csv}\n")
    return df


if __name__ == "__main__":
    b_dir = Path(__file__).resolve().parent / "benchmarks"
    o_csv = b_dir / "benchmark_results.csv"
    run_all_benchmarks(b_dir, o_csv)
