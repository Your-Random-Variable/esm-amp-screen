#!/usr/bin/env python3
"""
Docking of Designed AMPs against Vibrio cholerae BamA (AlphaFold Q9KPW0)
Target: Lateral Gate Seam (beta1-beta16, Loops L4/L6, POTRA5 Hinge)
Engine: ESM-AMP-Screen v1.4.0 (Cartesian Simplex + HADDOCK Physics + Calibrated 56D ML Scorer)
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esm_amp_screen.docking_engine import dock_single_peptide_geometric
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import analyze_cocomplex_contacts
from esm_amp_screen.ml_scorer import MLPeptideScorer
from esm_amp_screen.utils import logger, ensure_dir

# 1. Define AMP Variants
AMP_VARIANTS = {
    "WT": {
        "name": "WT (Baseline)",
        "sequence": "DIFCGETCAFIPCITHVPGTCSCKSKVCYFN",
        "mutations": "None (Wild-Type)",
        "charge": -0.21,
        "deepstamp_score": 0.8520,
        "role": "Natural Reference"
    },
    "Design_A": {
        "name": "Design A (Minimal)",
        "sequence": "KIFCGETCAFIPCITKVPGTCKCKSKVCYFN",
        "mutations": "D1K + H16K + S22K",
        "charge": +3.70,
        "deepstamp_score": 0.8920,
        "role": "High Stability / Low Complexity"
    },
    "Design_B": {
        "name": "Design B (Balanced Lead)",
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKSKVCYFN",
        "mutations": "D1K + A9K + H16K + S22K",
        "charge": +4.70,
        "deepstamp_score": 0.8962,
        "role": "Optimal Balance of Charge & Fold"
    },
    "Design_C": {
        "name": "Design C (High-Cationic)",
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKKKVCYFN",
        "mutations": "D1K + A9K + H16K + S22K + S25K",
        "charge": +5.70,
        "deepstamp_score": 0.8985,
        "role": "High Membrane Permeation"
    },
    "Design_D": {
        "name": "Design D (Max Score)",
        "sequence": "KIFCGKTCKFIPCITKVPGTCKCKKKVCYFN",
        "mutations": "D1K + E6K + A9K + H16K + S22K + S25K",
        "charge": +7.69,
        "deepstamp_score": 0.8989,
        "role": "Maximum Antibacterial Score"
    }
}

def main():
    receptor_pdb = Path("vc_bama_af_q9kpw0.pdb")
    if not receptor_pdb.exists():
        logger.error(f"Receptor {receptor_pdb} not found!")
        return

    out_base = Path("screening_results_vc_bama_af")
    ensure_dir(out_base)

    amp_structures_dir = Path("screening_results_multitarget/amp_structures")

    # Define Lateral Gate Pocket Specification on V. cholerae BamA (Q9KPW0)
    # Target center derived from the beta16 seam (Gln720, Val770, Trp769, Leu780, Phe794, Phe712, Trp206)
    pocket_spec = {
        "center_x": 4.88,
        "center_y": 21.11,
        "center_z": -0.39,
        "size_x": 38.0,
        "size_y": 38.0,
        "size_z": 38.0,
        "pocket_id": 1,
        "contact_residues": [
            f"A:{r}" for r in [
                205, 206, 208,
                710, 711, 712, 713, 714, 715, 720, 721, 722, 723,
                765, 767, 768, 769, 770, 771, 772, 773, 776, 778, 780,
                794, 795, 797, 798, 799, 800, 803
            ]
        ]
    }

    # Initialize calibrated ML Scorer
    model_weights = Path("data/trained_complex_ensemble.joblib")
    ml_scorer = MLPeptideScorer(model_path=model_weights if model_weights.exists() else None)
    logger.info(f"Initialized ML Scorer (calibrated: {ml_scorer.is_calibrated})")

    results_table = []

    print("\n" + "=" * 90)
    print("STARTING DOCKING ON VIBRIO CHOLERAE BAMA (ALPHAFOLD Q9KPW0) - LATERAL GATE")
    print("=" * 90)

    for pep_key, meta in AMP_VARIANTS.items():
        pep_pdb = amp_structures_dir / f"{pep_key}.pdb"
        if not pep_pdb.exists():
            logger.error(f"Peptide PDB {pep_pdb} not found!")
            continue

        pep_out_dir = out_base / pep_key
        ensure_dir(pep_out_dir)

        print(f"\n[Docking] Running native 3D geometric simplex docking for {meta['name']} ...")
        t0 = time.time()

        dock_res = dock_single_peptide_geometric(
            pep_id=pep_key,
            peptide_pdb=pep_pdb,
            receptor_pdb=receptor_pdb,
            pocket_spec=pocket_spec,
            out_dir=pep_out_dir,
            num_rotations=48,
            num_depths=8,
            num_modes=5,
            local_only=False
        )

        dt = time.time() - t0
        cpx_pdb = dock_res["complex_pdb"]
        dock_affinity = dock_res["best_affinity"]
        raw_score = dock_res["raw_score"]
        clashes = dock_res["clashes"]

        print(f"  -> Docking completed in {dt:.1f}s | Raw Score: {raw_score:.1f} | Affinity: {dock_affinity:.2f} kcal/mol | Clashes: {clashes}")

        # Physics Evaluation
        phys = calculate_haddock_physics_score(
            cpx_pdb,
            pocket_residues=pocket_spec["contact_residues"],
            receptor_chain="A",
            peptide_chain="B"
        )

        # Cross-chain Contact Analysis
        contacts = analyze_cocomplex_contacts(
            cpx_pdb,
            receptor_chain="A",
            peptide_chain="B"
        )

        # 56D ML Scoring
        ml_res = ml_scorer.score_complex(
            cpx_pdb,
            hit_meta={"affinity_kcal_mol": dock_affinity},
            pocket_residues=pocket_spec["contact_residues"]
        )

        ml_dG = ml_res.get("ml_delta_g", dock_affinity)
        ml_kd_str = ml_res.get("ml_kd", "N/A")
        consensus = ml_res.get("consensus_score", 0.0)

        results_table.append({
            "Variant": pep_key,
            "Name": meta["name"],
            "Sequence": meta["sequence"],
            "Charge": meta["charge"],
            "DeepSTAMP_P": meta["deepstamp_score"],
            "Dock_Affinity_kcal": dock_affinity,
            "HADDOCK_Score": phys["haddock_score"],
            "E_vdw": phys["e_vdw"],
            "E_elec": phys["e_elec"],
            "E_desolv": phys["e_desolv"],
            "BSA_A2": phys["bsa"],
            "H_Bonds": contacts["hbonds"],
            "Salt_Bridges": contacts["salt_bridges"],
            "Hydrophobic": contacts["hydrophobic_contacts"],
            "Clashes": contacts["clashes"],
            "ML_Delta_G": ml_dG,
            "ML_Kd": ml_kd_str,
            "Consensus_Score": round(consensus, 2),
            "Complex_PDB": str(cpx_pdb)
        })

    # Sort results by ML affinity
    df = pd.DataFrame(results_table)
    df = df.sort_values(by=["ML_Delta_G", "BSA_A2"], ascending=[True, False]).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    csv_path = out_base / "vc_bama_af_screening_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("FINAL SCREENING RESULTS: VIBRIO CHOLERAE BAMA (ALPHAFOLD Q9KPW0)")
    print("=" * 90)
    print(df[["Rank", "Name", "Charge", "DeepSTAMP_P", "Dock_Affinity_kcal", "HADDOCK_Score", "BSA_A2", "H_Bonds", "ML_Delta_G", "ML_Kd", "Consensus_Score"]].to_string(index=False))
    print("=" * 90)
    print(f"\nDetailed CSV exported to: {csv_path}")

if __name__ == "__main__":
    main()
