#!/usr/bin/env python3
"""
Multi-Target Virtual Screening Pipeline for Designed Antimicrobial Peptides (AMPs)
Evaluates WT, Design A, Design B, Design C, and Design D across:
  1. BamA (Lateral Gate Pocket #2) - Primary Bactericidal Potency (100% Essential)
  2. OmpU (Constriction Eyelet Pocket #1) - V. cholerae Specific Virulence Disarming
  3. LptD (Lateral LPS Gate Pocket #2) - LPS Assembly & Outer Membrane Biogenesis

Powered by ESM-AMP-Screen:
  - High-Resolution 7-Ray PSP LIGSITE Cavity Profiling
  - Cylindrical Normal Depth Offset Geometric Simplex Docking
  - Full-Atom HADDOCK 2.4 Physics (Evdw, Eelec, Edesolv, BSA)
  - 56-Feature Gradient Boosting ML Scorer (ML ΔG & Kd)
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import gemmi

from esm_amp_screen.esm_folder import fold_peptide
from esm_amp_screen.receptor_prep import detect_surface_cavities
from esm_amp_screen.docking_engine import dock_single_peptide_geometric
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import analyze_cocomplex_contacts, estimate_solution_affinity
from esm_amp_screen.ml_scorer import MLPeptideScorer
from esm_amp_screen.utils import logger, ensure_dir

# --- 1. Define AMP Variants ---
AMP_VARIANTS = {
    "WT": {
        "name": "WT (Baseline)",
        "sequence": "DIFCGETCAFIPCITHVPGTCSCKSKVCYFN",
        "mutations": "None (Wild-Type)",
        "charge": -0.21,
        "score": 0.8520,
        "uH": 0.535,
        "role": "Natural Reference"
    },
    "Design_A": {
        "name": "Design A (Minimal)",
        "sequence": "KIFCGETCAFIPCITKVPGTCKCKSKVCYFN",
        "mutations": "D1K + H16K + S22K (3 muts)",
        "charge": +3.70,
        "score": 0.8920,
        "uH": 0.400,
        "role": "High Stability / Low Complexity"
    },
    "Design_B": {
        "name": "Design B (Balanced)",
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKSKVCYFN",
        "mutations": "D1K + A9K + H16K + S22K (4 muts)",
        "charge": +4.70,
        "score": 0.8962,
        "uH": 0.216,
        "role": "Optimal Charge & Fold Balance"
    },
    "Design_C": {
        "name": "Design C (High-Cationic)",
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKKKVCYFN",
        "mutations": "D1K + A9K + H16K + S22K + S25K (5 muts)",
        "charge": +5.70,
        "score": 0.8985,
        "uH": 0.116,
        "role": "High Bacterial Permeation"
    },
    "Design_D": {
        "name": "Design D (Max Score)",
        "sequence": "KIFCGKTCKFIPCITKVPGTCKCKKKVCYFN",
        "mutations": "D1K + E6K + A9K + H16K + S22K + S25K (6 muts)",
        "charge": +7.69,
        "score": 0.8989,
        "uH": 0.103,
        "role": "Maximum Antibacterial Score"
    }
}

# --- 2. Define Bacterial Receptors ---
TARGETS = {
    "BamA": {
        "receptor_pdb": Path("bama_clean.pdb"),
        "name": "BamA (4N75)",
        "target_pocket_id": 2, # Lateral Gate beta1-beta16 seam
        "role": "Primary Bactericidal Potency (100% Essential)",
        "mechanism": "Binds lateral gate seam, blocks beta-barrel biogenesis -> rapid outer membrane collapse"
    },
    "OmpU": {
        "receptor_pdb": Path("ompu_clean.pdb"),
        "name": "OmpU (5ONU)",
        "target_pocket_id": 3, # Extracellular vestibule crown (Loop 3 / Loop 4 entrance)
        "role": "V. cholerae-Specific Disarming & Mucosal Clearance",
        "mechanism": "Caps virulence porin vestibule -> bile salt susceptibility & immune clearance"
    },
    "LptD": {
        "receptor_pdb": Path("lptd_clean.pdb"),
        "name": "LptD/LptE (4RHB)",
        "target_pocket_id": 2, # Lateral exit gate
        "role": "LPS Transport Disruption (Rank #2 Essential)",
        "mechanism": "Occludes lateral exit gate -> blocks LPS translocon to outer leaflet"
    }
}


def build_all_peptides(amp_dir: Path):
    """Folds all 5 AMPs to 3D PDB structures."""
    ensure_dir(amp_dir)
    peptide_paths = {}
    for pep_id, info in AMP_VARIANTS.items():
        pdb_path = amp_dir / f"{pep_id}.pdb"
        if not pdb_path.exists():
            print(f"Folding {pep_id} ({info['name']}) ...")
            fold_peptide(info["sequence"], pep_id, pdb_path, conformation="helix")
        peptide_paths[pep_id] = pdb_path
    return peptide_paths


def run_screening():
    base_out = Path("screening_results_multitarget")
    ensure_dir(base_out)
    amp_dir = base_out / "amp_structures"
    pep_paths = build_all_peptides(amp_dir)

    scorer = MLPeptideScorer()
    master_records = []

    print("\n==============================================================================================================")
    print("                     STARTING MULTI-TARGET VIRTUAL SCREENING ACROSS 3 ESSENTIAL TARGETS")
    print("==============================================================================================================\n")

    for target_key, tinfo in TARGETS.items():
        t_name = tinfo["name"]
        rec_pdb = tinfo["receptor_pdb"]
        target_pid = tinfo["target_pocket_id"]
        t_out = base_out / target_key
        ensure_dir(t_out)

        print(f"\n>>> TARGET: {t_name} | Role: {tinfo['role']}")
        print(f"    Detecting cavities on {rec_pdb.name}...")

        pockets = detect_surface_cavities(rec_pdb, max_pockets=5, padding=8.0, min_pocket_distance=10.0)
        pocket_dict = {p["pocket_id"]: p for p in pockets}

        if target_pid not in pocket_dict:
            target_pid = 1
        pkt = pocket_dict[target_pid]

        print(f"    Selected Pocket #{target_pid}: Druggability {pkt['druggability_score']:.1f}/100, Vol {pkt['volume_angstrom3']:.1f} Å³")
        print(f"    Center: ({pkt['center_x']:.1f}, {pkt['center_y']:.1f}, {pkt['center_z']:.1f})")
        print(f"    Lining Residues ({len(pkt['contact_residues'])}): {', '.join(pkt['contact_residues'][:6])}...")

        target_records = []

        for pep_id, pinfo in AMP_VARIANTS.items():
            pep_pdb = pep_paths[pep_id]
            run_dir = t_out / pep_id
            ensure_dir(run_dir)

            t0 = time.time()
            dock_res = dock_single_peptide_geometric(
                pep_id=pep_id,
                peptide_pdb=pep_pdb,
                receptor_pdb=rec_pdb,
                pocket_spec=pkt,
                out_dir=run_dir,
                num_rotations=24,
                num_depths=5,
                num_modes=3
            )
            elapsed = time.time() - t0

            cpx_pdb = dock_res["complex_pdb"]
            dock_aff = dock_res["best_affinity"]

            # Explicit HADDOCK Physics
            phys = calculate_haddock_physics_score(
                cpx_pdb,
                pocket_residues=pkt["contact_residues"],
                receptor_chain="A",
                peptide_chain="B"
            )

            # Interface contacts
            contacts = analyze_cocomplex_contacts(cpx_pdb, receptor_chain="A", peptide_chain="B")

            # ML Scoring
            hit_meta = {
                "pep_id": pep_id,
                "sequence": pinfo["sequence"],
                "affinity": dock_aff,
                "hbonds": contacts["hbonds"],
                "salt_bridges": contacts["salt_bridges"],
                "hydrophobic_contacts": contacts["hydrophobic_contacts"],
                "clashes": contacts["clashes"]
            }
            ml_res = scorer.score_complex(cpx_pdb, hit_meta=hit_meta, pocket_residues=pkt["contact_residues"])

            rec = {
                "Target": target_key,
                "Target_Name": t_name,
                "Target_Role": tinfo["role"],
                "Pocket_ID": target_pid,
                "Pocket_Vol": pkt["volume_angstrom3"],
                "Pocket_Drug": pkt["druggability_score"],
                "AMP_ID": pep_id,
                "AMP_Name": pinfo["name"],
                "Mutations": pinfo["mutations"],
                "Net_Charge": pinfo["charge"],
                "Antibacterial_Score": pinfo["score"],
                "Hydrophobic_Moment": pinfo["uH"],
                "Dock_Delta_G": dock_aff,
                "HADDOCK_Score": phys["haddock_score"],
                "E_vdw": phys["e_vdw"],
                "E_elec": phys["e_elec"],
                "E_desolv": phys["e_desolv"],
                "BSA": phys["bsa"],
                "H_Bonds": contacts["hbonds"],
                "Salt_Bridges": contacts["salt_bridges"],
                "Hydrophobic": contacts["hydrophobic_contacts"],
                "Clashes": contacts["clashes"],
                "ML_Delta_G": ml_res["ml_delta_g"],
                "ML_Kd": ml_res["ml_kd"],
                "Consensus_Score": ml_res["consensus_score"],
                "Complex_PDB": str(cpx_pdb),
                "Dock_Time_s": round(elapsed, 2)
            }
            target_records.append(rec)
            master_records.append(rec)
            print(f"      [✓] {pinfo['name']:<24}: Dock ΔG={dock_aff:>6.2f} kcal/mol | HADDOCK={phys['haddock_score']:>7.2f} | BSA={phys['bsa']:>5.1f} Å² | ML ΔG={ml_res['ml_delta_g']:>6.2f} | Kd={ml_res['ml_kd']:<8} | Clashes={contacts['clashes']}")

        df_target = pd.DataFrame(target_records)
        df_target.sort_values(by="Consensus_Score", ascending=False, inplace=True)
        df_target.to_csv(t_out / f"{target_key}_leaderboard.csv", index=False)

    df_master = pd.DataFrame(master_records)
    df_master.to_csv(base_out / "multitarget_master_leaderboard.csv", index=False)

    # Print nicely formatted summary tables
    print("\n" + "=" * 135)
    print("                    ESM-AMP-SCREEN: MULTI-TARGET VIRTUAL SCREENING SUMMARY RESULTS")
    print("=" * 135)

    for target_key in TARGETS.keys():
        sub_df = df_master[df_master["Target"] == target_key].copy()
        sub_df.sort_values(by="Consensus_Score", ascending=False, inplace=True)
        t_meta = TARGETS[target_key]

        print(f"\n--- TARGET: {t_meta['name']} ({t_meta['role']}) | Pocket #{t_meta['target_pocket_id']} ---")
        hdr = f"{'Rank':<5} {'AMP Design':<24} {'Charge':<8} {'Dock ΔG':<10} {'HADDOCK':<10} {'BSA (Å²)':<10} {'ML ΔG':<10} {'Pred Kd':<12} {'Contacts (HB/SB/HP)':<22} {'Consensus':<10}"
        print(hdr)
        print("-" * len(hdr))

        for rank, (_, row) in enumerate(sub_df.iterrows(), 1):
            cont = f"{row['H_Bonds']}HB / {row['Salt_Bridges']}SB / {row['Hydrophobic']}HP"
            print(f"{rank:<5} {row['AMP_Name']:<24} {row['Net_Charge']:<+8.1f} {row['Dock_Delta_G']:<10.2f} {row['HADDOCK_Score']:<10.2f} {row['BSA']:<10.1f} {row['ML_Delta_G']:<10.2f} {str(row['ML_Kd']):<12} {cont:<22} {row['Consensus_Score']:<10.2f}")

    print("\n" + "=" * 135)
    print("                      OVERALL MULTI-TARGET SUMMARY & CROSS-TARGET SELECTIVITY")
    print("=" * 135)

    piv_consensus = df_master.pivot(index="AMP_Name", columns="Target", values="Consensus_Score")
    piv_mldg = df_master.pivot(index="AMP_Name", columns="Target", values="ML_Delta_G")
    piv_bsa = df_master.pivot(index="AMP_Name", columns="Target", values="BSA")

    print("\n[Consensus Score by Target (Higher is Better)]")
    print(piv_consensus.to_string())

    print("\n[ML ΔG Binding Affinity (kcal/mol, More Negative is Tighter)]")
    print(piv_mldg.to_string())

    print("\n[Buried Surface Area (Å²)]")
    print(piv_bsa.to_string())

    print("\n[OK] Virtual screening complete! All data stored in: screening_results_multitarget/")


if __name__ == "__main__":
    run_screening()
