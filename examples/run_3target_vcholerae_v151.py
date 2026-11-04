#!/usr/bin/env python3
"""
ESM-AMP-Screen v1.5.1 — Corrected 3-Target V. cholerae Virtual Screening
==========================================================================
Evaluates WT, Design A, Design B, Design C, Design D across:
  1. BamA  (V. cholerae AlphaFold AF-Q9KPW0) — Lateral gate β1-β16 seam
  2. OmpU  (PDB 5ONU)                          — Extracellular vestibule crown
  3. LptD  (PDB 4RHB)                          — Lateral LPS exit gate

Key corrections vs. old run_multitarget_screen.py:
  • BamA receptor is now vc_bama_af_q9kpw0.pdb (Q9KPW0, 803 res), NOT 4N75 E. coli
  • BamA pocket is specified manually (pre-validated center & box)
  • OmpU/LptD use LIGSITE auto-detect (pocket_id 3 and 2 respectively)
  • Pre-built AMP PDBs reused from screening_results_multitarget/amp_structures/
  • Output written to screening_results_3target_v151/

Usage:
  /home/lab202/miniconda3/envs/deepstamp_env/bin/python \\
      examples/run_3target_vcholerae_v151.py 2>&1 | tee run_3target_v151.log
"""

import sys
import time
from pathlib import Path

import pandas as pd

from esm_amp_screen.receptor_prep import detect_surface_cavities
from esm_amp_screen.docking_engine import dock_single_peptide_geometric
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import analyze_cocomplex_contacts
from esm_amp_screen.ml_scorer import MLPeptideScorer
from esm_amp_screen.utils import logger, ensure_dir

# ---------------------------------------------------------------------------
# 1.  AMP Variants — exact sequences, charges, DeepSTAMP scores
# ---------------------------------------------------------------------------
AMP_VARIANTS = {
    "WT": {
        "name": "WT (Baseline)",
        "sequence": "DIFCGETCAFIPCITHVPGTCSCKSKVCYFN",
        "mutations": "None (Wild-Type)",
        "charge": -0.21,
        "score": 0.8520,
        "uH": 0.535,
        "role": "Natural Reference",
    },
    "Design_A": {
        "name": "Design A (Minimal)",
        "sequence": "KIFCGETCAFIPCITKVPGTCKCKSKVCYFN",
        "mutations": "D1K + H16K + S22K (3 muts)",
        "charge": +3.70,
        "score": 0.8920,
        "uH": 0.400,
        "role": "High Stability / Low Complexity",
    },
    "Design_B": {
        "name": "Design B (Balanced)",
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKSKVCYFN",
        "mutations": "D1K + A9K + H16K + S22K (4 muts)",
        "charge": +4.70,
        "score": 0.8962,
        "uH": 0.216,
        "role": "Optimal Charge & Fold Balance",
    },
    "Design_C": {
        "name": "Design C (High-Cationic)",
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKKKVCYFN",
        "mutations": "D1K + A9K + H16K + S22K + S25K (5 muts)",
        "charge": +5.70,
        "score": 0.8985,
        "uH": 0.116,
        "role": "High Bacterial Permeation",
    },
    "Design_D": {
        "name": "Design D (Max Score)",
        "sequence": "KIFCGKTCKFIPCITKVPGTCKCKKKVCYFN",
        "mutations": "D1K + E6K + A9K + H16K + S22K + S25K (6 muts)",
        "charge": +7.69,
        "score": 0.8989,
        "uH": 0.103,
        "role": "Maximum Antibacterial Score",
    },
}

# ---------------------------------------------------------------------------
# 2.  Targets — corrected receptors & pocket specs
# ---------------------------------------------------------------------------
#
# BamA: manual pocket spec (pre-validated from screening_results_vc_bama_af/)
#   center=(4.88, 21.11, -0.39), box=38×38×38 Å — lateral gate β1-β16 seam
#
# OmpU/LptD: LIGSITE auto-detect (pocket_id referenced from earlier run)
# ---------------------------------------------------------------------------
TARGETS = {
    "BamA": {
        "receptor_pdb": Path("vc_bama_af_q9kpw0.pdb"),
        "name": "BamA — V. cholerae AF Q9KPW0",
        "pocket_mode": "manual",
        "pocket_spec": {
            "pocket_id": 1,
            "center_x": 4.88,
            "center_y": 21.11,
            "center_z": -0.39,
            "size_x": 38.0,
            "size_y": 38.0,
            "size_z": 38.0,
            "contact_residues": [
                f"A:{r}" for r in [
                    205, 206, 208,
                    710, 711, 712, 713, 714, 715, 720, 721, 722, 723,
                    765, 767, 768, 769, 770, 771, 772, 773, 776, 778, 780,
                    794, 795, 797, 798, 799, 800, 803
                ]
            ],
            "volume_angstrom3": 4200.0,
            "druggability_score": 72.0,
        },
        "docking_params": {
            "num_rotations": 48,
            "num_depths": 8,
            "num_modes": 5,
        },
        "role": "Primary Bactericidal Potency (100% Essential)",
        "mechanism": (
            "Binds lateral gate seam, blocks β-barrel biogenesis "
            "→ rapid outer membrane collapse"
        ),
    },
    "OmpU": {
        "receptor_pdb": Path("ompu_clean.pdb"),
        "name": "OmpU — V. cholerae (PDB 5ONU)",
        "pocket_mode": "auto",
        "target_pocket_id": 3,
        "docking_params": {
            "num_rotations": 24,
            "num_depths": 5,
            "num_modes": 3,
        },
        "role": "V. cholerae-Specific Disarming & Mucosal Clearance",
        "mechanism": (
            "Caps virulence porin vestibule "
            "→ bile-salt susceptibility & immune clearance"
        ),
    },
    "LptD": {
        "receptor_pdb": Path("lptd_clean.pdb"),
        "name": "LptD/LptE — (PDB 4RHB)",
        "pocket_mode": "auto",
        "target_pocket_id": 2,
        "docking_params": {
            "num_rotations": 24,
            "num_depths": 5,
            "num_modes": 3,
        },
        "role": "LPS Transport Disruption (Rank #2 Essential)",
        "mechanism": (
            "Occludes lateral exit gate "
            "→ blocks LPS translocon to outer leaflet"
        ),
    },
}


# ---------------------------------------------------------------------------
# 3.  Helper — resolve pocket specification for a target
# ---------------------------------------------------------------------------
def get_pocket(tinfo: dict) -> dict:
    """Return a pocket dict ready for dock_single_peptide_geometric."""
    if tinfo["pocket_mode"] == "manual":
        return tinfo["pocket_spec"]

    # Auto-detect via LIGSITE
    pockets = detect_surface_cavities(
        tinfo["receptor_pdb"], max_pockets=5, padding=8.0, min_pocket_distance=10.0
    )
    pocket_dict = {p["pocket_id"]: p for p in pockets}
    target_pid = tinfo["target_pocket_id"]
    if target_pid not in pocket_dict:
        logger.warning(
            f"Pocket #{target_pid} not found on {tinfo['receptor_pdb'].name}; "
            "falling back to pocket #1"
        )
        target_pid = 1
    return pocket_dict[target_pid]


# ---------------------------------------------------------------------------
# 4.  Main screening loop
# ---------------------------------------------------------------------------
def run_screening():
    base_out = Path("screening_results_3target_v151")
    ensure_dir(base_out)

    # Pre-built AMP PDB structures (already folded, skip re-folding)
    amp_dir = Path("screening_results_multitarget") / "amp_structures"
    if not amp_dir.exists():
        print(
            f"[ERROR] AMP structures directory not found: {amp_dir}\n"
            "Run the original screening first to generate AMP PDBs.",
            file=sys.stderr,
        )
        sys.exit(1)

    pep_paths = {}
    for pep_id in AMP_VARIANTS:
        p = amp_dir / f"{pep_id}.pdb"
        if not p.exists():
            print(f"[ERROR] Missing AMP PDB: {p}", file=sys.stderr)
            sys.exit(1)
        pep_paths[pep_id] = p

    # ML scorer — loads pre-trained ensemble from data/
    model_weights = Path("data/trained_complex_ensemble.joblib")
    scorer = MLPeptideScorer(model_path=model_weights if model_weights.exists() else None)

    master_records = []

    print("\n" + "=" * 110)
    print("   ESM-AMP-Screen v1.5.1 — 3-Target V. cholerae Virtual Screening")
    print("   BamA (AF Q9KPW0) | OmpU (5ONU) | LptD (4RHB)")
    print("=" * 110 + "\n")

    for target_key, tinfo in TARGETS.items():
        t_name = tinfo["name"]
        rec_pdb = tinfo["receptor_pdb"]
        t_out = base_out / target_key
        ensure_dir(t_out)

        if not rec_pdb.exists():
            print(f"[ERROR] Receptor not found: {rec_pdb}", file=sys.stderr)
            sys.exit(1)

        print(f"\n>>> TARGET: {t_name}")
        print(f"    Role   : {tinfo['role']}")
        print(f"    Pocket : ", end="", flush=True)

        pkt = get_pocket(tinfo)

        # For BamA manual pocket, populate contact_residues from receptor structure
        if tinfo["pocket_mode"] == "manual" and not pkt["contact_residues"]:
            # Use all residues within 12 Å of pocket center as lining residues
            import gemmi
            st = gemmi.read_structure(str(rec_pdb))
            cx, cy, cz = pkt["center_x"], pkt["center_y"], pkt["center_z"]
            lining = []
            for model in st:
                for chain in model:
                    for res in chain:
                        for atom in res:
                            dx = atom.pos.x - cx
                            dy = atom.pos.y - cy
                            dz = atom.pos.z - cz
                            if (dx**2 + dy**2 + dz**2) ** 0.5 < 12.0:
                                rid = f"{res.name}{res.seqid.num}{chain.name}"
                                if rid not in lining:
                                    lining.append(rid)
                                break  # one atom per residue is enough
            pkt["contact_residues"] = lining

        print(
            f"center=({pkt['center_x']:.1f}, {pkt['center_y']:.1f}, "
            f"{pkt['center_z']:.1f}), "
            f"vol={pkt.get('volume_angstrom3', 0):.0f} Å³, "
            f"drug={pkt.get('druggability_score', 0):.1f}/100"
        )
        print(
            f"    Lining : {len(pkt['contact_residues'])} residues — "
            f"{', '.join(pkt['contact_residues'][:6])}..."
        )

        target_records = []

        for pep_id, pinfo in AMP_VARIANTS.items():
            pep_pdb = pep_paths[pep_id]
            run_dir = t_out / pep_id
            ensure_dir(run_dir)

            dparams = tinfo.get("docking_params", {"num_rotations": 24, "num_depths": 5, "num_modes": 3})
            t0 = time.time()
            dock_res = dock_single_peptide_geometric(
                pep_id=pep_id,
                peptide_pdb=pep_pdb,
                receptor_pdb=rec_pdb,
                pocket_spec=pkt,
                out_dir=run_dir,
                num_rotations=dparams["num_rotations"],
                num_depths=dparams["num_depths"],
                num_modes=dparams["num_modes"],
                local_only=False,
            )
            elapsed = time.time() - t0

            cpx_pdb = dock_res["complex_pdb"]
            dock_aff = dock_res["best_affinity"]

            # HADDOCK physics score
            phys = calculate_haddock_physics_score(
                cpx_pdb,
                pocket_residues=pkt["contact_residues"],
                receptor_chain="A",
                peptide_chain="B",
            )

            # Interface contact analysis
            contacts = analyze_cocomplex_contacts(
                cpx_pdb, receptor_chain="A", peptide_chain="B"
            )

            # ML scoring
            hit_meta = {
                "pep_id": pep_id,
                "sequence": pinfo["sequence"],
                "affinity": dock_aff,
                "hbonds": contacts["hbonds"],
                "salt_bridges": contacts["salt_bridges"],
                "hydrophobic_contacts": contacts["hydrophobic_contacts"],
                "clashes": contacts["clashes"],
            }
            ml_res = scorer.score_complex(
                cpx_pdb,
                hit_meta=hit_meta,
                pocket_residues=pkt["contact_residues"],
            )

            rec = {
                "Target": target_key,
                "Target_Name": t_name,
                "Target_Role": tinfo["role"],
                "Pocket_Center": (
                    f"({pkt['center_x']:.1f},{pkt['center_y']:.1f},{pkt['center_z']:.1f})"
                ),
                "Pocket_Vol_A3": round(pkt.get("volume_angstrom3", 0), 1),
                "Pocket_Drug": round(pkt.get("druggability_score", 0), 1),
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
                "BSA_A2": phys["bsa"],
                "H_Bonds": contacts["hbonds"],
                "Salt_Bridges": contacts["salt_bridges"],
                "Hydrophobic": contacts["hydrophobic_contacts"],
                "Clashes": contacts["clashes"],
                "ML_Delta_G": ml_res["ml_delta_g"],
                "ML_Kd": ml_res["ml_kd"],
                "Consensus_Score": ml_res["consensus_score"],
                "Complex_PDB": str(cpx_pdb),
                "Dock_Time_s": round(elapsed, 2),
            }
            target_records.append(rec)
            master_records.append(rec)

            clash_flag = " ⚠ CLASH" if contacts["clashes"] > 0 else ""
            print(
                f"      [✓] {pinfo['name']:<24}: "
                f"Dock ΔG={dock_aff:>6.2f} kcal/mol | "
                f"HADDOCK={phys['haddock_score']:>7.2f} | "
                f"BSA={phys['bsa']:>5.1f} Å² | "
                f"ML ΔG={ml_res['ml_delta_g']:>6.2f} | "
                f"Kd={str(ml_res['ml_kd']):<10} | "
                f"Clashes={contacts['clashes']}"
                f"{clash_flag}"
            )

        df_target = pd.DataFrame(target_records)
        df_target.sort_values("Consensus_Score", ascending=False, inplace=True)
        df_target.to_csv(t_out / f"{target_key}_leaderboard.csv", index=False)

    # -----------------------------------------------------------------------
    # 5.  Unified master CSV
    # -----------------------------------------------------------------------
    df_master = pd.DataFrame(master_records)
    out_csv = base_out / "multitarget_master_leaderboard_v151.csv"
    df_master.to_csv(out_csv, index=False)

    # -----------------------------------------------------------------------
    # 6.  Pretty-print summary tables
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 125)
    print("       ESM-AMP-Screen v1.5.1 — MULTI-TARGET RESULTS (V. cholerae; BamA = AF Q9KPW0)")
    print("=" * 125)

    for target_key, tinfo in TARGETS.items():
        sub = df_master[df_master["Target"] == target_key].copy()
        sub.sort_values("Consensus_Score", ascending=False, inplace=True)

        print(f"\n--- {tinfo['name']} | {tinfo['role']} ---")
        hdr = (
            f"{'Rk':<3} {'AMP Design':<24} {'Chg':>5} "
            f"{'Dock ΔG':>8} {'HADDOCK':>9} {'BSA(Å²)':>8} "
            f"{'ML ΔG':>7} {'Pred Kd':<11} "
            f"{'HB/SB/HP':<10} {'Clash':>5} {'Consensus':>9}"
        )
        print(hdr)
        print("-" * len(hdr))
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            cont = f"{int(row['H_Bonds'])}HB/{int(row['Salt_Bridges'])}SB/{int(row['Hydrophobic'])}HP"
            clash_str = f"{int(row['Clashes'])}⚠" if row["Clashes"] > 0 else "0"
            print(
                f"{rank:<3} {row['AMP_Name']:<24} {row['Net_Charge']:>+5.1f} "
                f"{row['Dock_Delta_G']:>8.2f} {row['HADDOCK_Score']:>9.2f} {row['BSA_A2']:>8.1f} "
                f"{row['ML_Delta_G']:>7.2f} {str(row['ML_Kd']):<11} "
                f"{cont:<10} {clash_str:>5} {row['Consensus_Score']:>9.3f}"
            )

    # Cross-target pivot
    print("\n\n" + "=" * 80)
    print("   CROSS-TARGET CONSENSUS SCORE (Higher = Better)")
    print("=" * 80)
    piv = df_master.pivot(index="AMP_Name", columns="Target", values="Consensus_Score")
    piv["Multi_Target_Mean"] = piv.mean(axis=1)
    piv.sort_values("Multi_Target_Mean", ascending=False, inplace=True)
    print(piv.round(3).to_string())

    print("\n\n" + "=" * 80)
    print("   CROSS-TARGET ML ΔG (kcal/mol, More Negative = Tighter Binding)")
    print("=" * 80)
    piv_dg = df_master.pivot(index="AMP_Name", columns="Target", values="ML_Delta_G")
    piv_dg["Mean_ML_dG"] = piv_dg.mean(axis=1)
    piv_dg.sort_values("Mean_ML_dG", inplace=True)
    print(piv_dg.round(2).to_string())

    print(f"\n[OK] Screening complete. Results in: {base_out.resolve()}/")
    print(f"[OK] Master CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    run_screening()
