"""
Command-Line Interface (CLI) for ESM-AMP-Screen.
Ultra-fast, ESM-powered Protein-Peptide & AMP Virtual Screening Pipeline.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import gemmi

from esm_amp_screen import __version__
from esm_amp_screen.utils import logger, read_fasta, ensure_dir, which_tool, AA_3TO1
from esm_amp_screen.esm_folder import batch_fold_peptides, fetch_rcsb_pdb, predict_secondary_structure_propensity
from esm_amp_screen.receptor_prep import clean_receptor_pdb, calculate_grid_box, detect_surface_cavities
from esm_amp_screen.peptide_prep import batch_prepare_peptides, prepare_native_peptides, create_docked_pristine_pdb
from esm_amp_screen.docking_engine import run_parallel_screening
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import (
    calculate_net_charge,
    calculate_hydrophobic_moment,
    calculate_mean_hydrophobicity,
    calculate_boman_index,
    analyze_complex_contacts,
    compute_composite_score,
    count_heavy_atoms,
    calculate_ligand_efficiency,
    calculate_per_residue_energy,
    calculate_conformational_entropy_penalty,
    predict_hemolytic_risk_and_selectivity,
    estimate_solution_affinity
)
from esm_amp_screen.reporter import (
    export_screening_report,
    create_cocomplex_pdb,
    generate_pockets_pymol_script
)
from esm_amp_screen.ml_scorer import MLPeptideScorer


def run_multi_pocket_blind_docking(
    args: argparse.Namespace,
    rec_pdb_path: Path,
    detected_pockets: List[Dict[str, Any]],
    peptide_pdbs: List[Path],
    sequences: Dict[str, str],
    out_dir: Path,
    obabel_bin: Optional[str] = None
) -> None:
    """
    Executes comprehensive multi-pocket blind docking across the top N predicted druggable pockets:
    1. Evaluates all N distinct surface cavities detected on the target receptor.
    2. Docks peptide candidates into each pocket independently.
    3. Evaluates binding poses with HADDOCK explicit physics, solution thermodynamics, and ML consensus scorer.
    4. Produces a comparative multi-pocket leaderboard, CSV summary, and color-coded PyMOL visualization.
    """
    import pandas as pd
    from esm_amp_screen.physics_engine import calculate_haddock_physics_score
    from esm_amp_screen.ml_scorer import MLPeptideScorer
    from esm_amp_screen.reporter import generate_pockets_pymol_script

    print("\n" + "="*125)
    print(f"      MULTI-POCKET BLIND DOCKING: SCREENING ACROSS {len(detected_pockets)} DRUGGABLE RECEPTOR POCKETS")
    print("      Evaluating Spatial Druggability, Binding Affinities, HADDOCK Scores & Allosteric Cavities")
    print("="*125 + "\n")

    peptides_pdb_dir = out_dir / "peptides_pdb"
    ligand_paths = prepare_native_peptides(peptide_pdbs, peptides_pdb_dir)

    model_file = getattr(args, "ml_model", None)
    if model_file is None:
        default_pkg_model = Path(__file__).resolve().parent.parent / "data" / "trained_complex_ensemble.joblib"
        if default_pkg_model.exists():
            model_file = default_pkg_model
    ml_scorer = MLPeptideScorer(model_path=model_file)

    cpx_dir = out_dir / "top_complexes"
    ensure_dir(cpx_dir)

    pocket_records = []
    docked_complex_map = {}

    for i, p in enumerate(detected_pockets):
        p_id = p.get("pocket_id", i + 1)
        p_grid = {
            "center_x": p["center_x"], "center_y": p["center_y"], "center_z": p["center_z"],
            "size_x": p["size_x"], "size_y": p["size_y"], "size_z": p["size_z"]
        }
        p_res = p.get("contact_residues", [])
        pocket_dir = out_dir / f"pocket_{p_id}"
        dock_poses_dir = pocket_dir / "docking_poses"
        ensure_dir(dock_poses_dir)

        logger.info(
            f"Screening Pocket #{p_id}: Druggability={p['druggability_score']}/100, "
            f"Vol={p['volume_angstrom3']} Å³, Center=({p['center_x']}, {p['center_y']}, {p['center_z']})"
        )

        dock_res = run_parallel_screening(
            ligands=ligand_paths,
            receptor_pdb=rec_pdb_path,
            grid=p_grid,
            out_dir=dock_poses_dir,
            threads=args.threads,
            max_workers=args.max_workers,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes
        )

        successful = [d for d in dock_res.values() if d.get("success")]
        if not successful:
            logger.warning(f"No successful poses found for Pocket #{p_id}.")
            continue

        successful.sort(key=lambda x: x.get("best_affinity", 0.0))
        best_hit = successful[0]
        pep_id = best_hit["pep_id"]
        best_aff = best_hit["best_affinity"]
        best_pose = Path(best_hit["docked_pdb"])

        cpx_pdb = cpx_dir / f"pocket_{p_id}_{pep_id}_complex.pdb"
        create_cocomplex_pdb(rec_pdb_path, best_pose, cpx_pdb)
        docked_complex_map[p_id] = cpx_pdb

        phys = calculate_haddock_physics_score(
            cpx_pdb,
            pocket_residues=p_res,
            receptor_chain=getattr(args, "rec_chain", "A"),
            peptide_chain="B"
        )
        seq = sequences.get(pep_id, "")
        sol_aff = estimate_solution_affinity(
            affinity_kcal_mol=best_aff,
            sequence=seq,
            conformation=getattr(args, "conformation", "auto"),
            is_cyclic=getattr(args, "disulfide_lock", True)
        )
        ml_res = ml_scorer.score_complex(
            complex_pdb=cpx_pdb,
            hit_meta={"pep_id": pep_id, "sequence": seq, "affinity": best_aff},
            pocket_residues=p_res
        )

        pocket_records.append({
            "pocket_id": p_id,
            "druggability_score": p["druggability_score"],
            "druggability_class": p["druggability_class"],
            "volume_angstrom3": p["volume_angstrom3"],
            "enclosure_score": p["enclosure_score"],
            "hydrophobic_ratio": p["hydrophobic_ratio"],
            "center_x": p["center_x"],
            "center_y": p["center_y"],
            "center_z": p["center_z"],
            "size_x": p["size_x"],
            "size_y": p["size_y"],
            "size_z": p["size_z"],
            "num_lining_residues": len(p_res),
            "contact_residues": ", ".join(p_res[:10]),
            "best_peptide": pep_id,
            "docking_delta_g": best_aff,
            "haddock_score": round(phys["haddock_score"], 2),
            "e_vdw": round(phys["e_vdw"], 2),
            "e_elec": round(phys["e_elec"], 2),
            "e_desolv": round(phys["e_desolv"], 2),
            "bsa": round(phys["bsa"], 1),
            "solution_delta_g": round(sol_aff["solution_delta_g"], 2),
            "solution_kd": sol_aff["solution_kd"],
            "ml_delta_g": round(ml_res["ml_delta_g"], 2),
            "ml_kd": ml_res["ml_kd"],
            "consensus_score": round(ml_res["consensus_score"], 2),
            "complex_pdb": str(cpx_pdb)
        })

    if not pocket_records:
        logger.error("No valid pocket docking results obtained.")
        return

    # Rank pockets by binding affinity (lowest Delta G) and consensus score
    pocket_records.sort(key=lambda x: (x["docking_delta_g"], -x["consensus_score"]))
    best_aff = pocket_records[0]["docking_delta_g"]

    for r in pocket_records:
        diff = r["docking_delta_g"] - best_aff
        if diff <= 0.5:
            r["pocket_role"] = "Primary Binding Site (Preferred)"
        elif diff <= 2.5:
            r["pocket_role"] = "Secondary Allosteric Site"
        else:
            r["pocket_role"] = "Non-Productive / Shallow Groove"

    # Print Multi-Pocket Comparison Leaderboard
    print("\n" + "="*145)
    print("                  MULTI-POCKET BLIND DOCKING LEADERBOARD (RANKED BY BINDING AFFINITY & CONSENSUS SCORE)")
    print("="*145)
    header = (
        f"{'Pocket':<8} {'Druggability':<14} {'Vol (Å³)':<10} {'Best Ligand':<16} "
        f"{'Dock ΔG':<10} {'HADDOCK':<10} {'BSA (Å²)':<10} {'ML ΔG':<10} {'Pred Kd':<10} {'Sol. Kd':<12} {'Role / Assessment':<32}"
    )
    print(header)
    print("-" * len(header))
    for r in pocket_records:
        p_str = f"P#{r['pocket_id']}"
        d_str = f"{r['druggability_score']:.1f}/100"
        vol_str = f"{r['volume_angstrom3']:.1f}"
        dock_str = f"{r['docking_delta_g']:.2f}"
        had_str = f"{r['haddock_score']:.1f}"
        bsa_str = f"{r['bsa']:.1f}"
        ml_str = f"{r['ml_delta_g']:.2f}"
        pred_str = str(r['ml_kd'])
        sol_str = str(r['solution_kd'])
        role_str = r['pocket_role']
        print(
            f"{p_str:<8} {d_str:<14} {vol_str:<10} {r['best_peptide'][:14]:<16} "
            f"{dock_str:<10} {had_str:<10} {bsa_str:<10} {ml_str:<10} {pred_str:<10} {sol_str:<12} {role_str:<32}"
        )
    print("="*145 + "\n")

    # Export CSV
    df = pd.DataFrame(pocket_records)
    csv_out1 = out_dir / "blind_docking_5_pockets.csv"
    csv_out2 = out_dir / "blind_docking_pockets.csv"
    df.to_csv(csv_out1, index=False)
    df.to_csv(csv_out2, index=False)

    # Export PyMOL visualizer
    pml_out = out_dir / "view_pockets.pml"
    generate_pockets_pymol_script(
        receptor_pdb=rec_pdb_path,
        pockets=detected_pockets,
        script_path=pml_out,
        docked_complexes=docked_complex_map
    )

    logger.info(f"Multi-pocket blind docking results saved to: {csv_out1}")
    logger.info(f"Color-coded PyMOL multi-pocket visualizer:   {pml_out}")
    logger.info(f"Top pocket complex structures saved to:      {cpx_dir}\n")


def run_pipeline(args: argparse.Namespace) -> None:
    """End-to-end automated screening workflow."""
    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)

    blind_k = getattr(args, "blind_pockets", 1)
    if getattr(args, "blind_docking", False) and blind_k <= 1:
        args.blind_pockets = 5
        blind_k = 5

    # 1. External tools verification
    obabel_bin = args.obabel_bin or which_tool("obabel")

    print("\n" + "="*70)
    print("      ESM-AMP-SCREEN: High-Throughput Protein-Peptide Virtual Screening")
    print("="*70 + "\n")

    # 2. Receptor resolution
    receptor_arg = args.receptor
    rec_chain = getattr(args, "rec_chain", "A")
    rec_pdb_path = out_dir / "target_receptor.pdb"
    if Path(receptor_arg).exists():
        rec_pdb_path = clean_receptor_pdb(Path(receptor_arg), rec_pdb_path, receptor_chain=rec_chain)
    elif len(receptor_arg) == 4 and receptor_arg.isalnum():
        # RCSB PDB ID
        fetch_rcsb_pdb(receptor_arg, rec_pdb_path)
        rec_pdb_path = clean_receptor_pdb(rec_pdb_path, out_dir / "target_receptor_clean.pdb", receptor_chain=rec_chain)
    else:
        raise FileNotFoundError(f"Receptor file not found and not a valid 4-letter PDB ID: {receptor_arg}")

    # 3. Calculate Grid Box / Pocket Geometry
    center_coords = tuple(map(float, args.center)) if args.center else None
    size_dims = tuple(map(float, args.size)) if args.size else None
    pocket_res = [r.strip() for r in args.pocket_residues.split(",")] if args.pocket_residues else None

    grid = calculate_grid_box(
        rec_pdb_path,
        center=center_coords,
        size=size_dims,
        pocket_residues=pocket_res,
        auto_pocket=args.auto_pocket,
        padding=args.padding,
        max_pockets=max(5, blind_k)
    )

    # 4. Load / Fold AMP Peptides
    library_arg = Path(args.library)
    peptides_pdb_dir = out_dir / "peptides_pdb"
    ensure_dir(peptides_pdb_dir)

    if library_arg.is_dir():
        # User already provided a folder of ESM-predicted PDB files
        logger.info(f"Using pre-generated peptide PDBs from directory: {library_arg}")
        peptide_pdbs = list(library_arg.glob("*.pdb"))
        sequences = {}
        for pdb_file in peptide_pdbs:
            pep_id = pdb_file.stem
            try:
                st = gemmi.read_structure(str(pdb_file))
                seq = "".join(
                    AA_3TO1.get(res.name.upper(), "X")
                    for m in st for c in m for res in c
                    if res.name.upper() in AA_3TO1
                )
                if seq:
                    sequences[pep_id] = seq
            except Exception as e:
                logger.warning(f"Could not extract sequence from {pdb_file.name}: {e}")
    elif library_arg.is_file():
        sequences = read_fasta(library_arg)
        folded = batch_fold_peptides(
            sequences,
            peptides_pdb_dir,
            conformation=args.conformation,
            hf_token=args.hf_token
        )
        peptide_pdbs = [p[0] for p in folded.values()]
    else:
        raise FileNotFoundError(f"Peptide library not found: {library_arg}")

    if not peptide_pdbs:
        logger.error("No peptide PDB structures available for screening.")
        return

    # Multi-Pocket Blind Docking Mode
    if blind_k > 1 and not center_coords and not pocket_res:
        target_pockets = grid.get("detected_pockets", [])
        if not target_pockets:
            target_pockets = detect_surface_cavities(rec_pdb_path, max_pockets=blind_k, padding=args.padding)
        if len(target_pockets) > 1:
            run_multi_pocket_blind_docking(
                args=args,
                rec_pdb_path=rec_pdb_path,
                detected_pockets=target_pockets[:blind_k],
                peptide_pdbs=peptide_pdbs,
                sequences=sequences,
                out_dir=out_dir,
                obabel_bin=obabel_bin
            )
            return

    # Stage 1: High-Throughput Screen (Fast Filter)
    print("\n" + "="*70)
    print("      STAGE 1: HIGH-THROUGHPUT VIRTUAL SCREENING (FAST FILTER)")
    print("="*70 + "\n")

    # 5. Prepare Native 3D Peptide PDBs
    ligand_paths = prepare_native_peptides(peptide_pdbs, peptides_pdb_dir)

    # 6. Run High-Throughput Native 3D Geometric Screening
    docking_out_dir = out_dir / "docking_poses"
    dock_results = run_parallel_screening(
        ligands=ligand_paths,
        receptor_pdb=rec_pdb_path,
        grid=grid,
        out_dir=docking_out_dir,
        threads=args.threads,
        max_workers=args.max_workers,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes
    )

    # 7. Biophysical Profiling & Contact Analysis for Stage 1
    logger.info("Computing AMP biophysical metrics and intermolecular contact fingerprints...")
    summary_data = []

    for pep_id, dres in dock_results.items():
        if not dres["success"]:
            continue

        seq = sequences.get(pep_id, "")
        affinity = dres["best_affinity"]
        top_pose_pdb = dres.get("docked_pdb")
        if not top_pose_pdb or not Path(top_pose_pdb).exists():
            top_pose_pdb = docking_out_dir / f"{pep_id}_pose1.pdb"

        # Calculate AMP metrics
        z = calculate_net_charge(seq) if seq else 0.0
        mu_h = calculate_hydrophobic_moment(seq, angle_deg=100.0 if args.conformation == "helix" else 160.0) if seq else 0.0
        mean_h = calculate_mean_hydrophobicity(seq) if seq else 0.0
        boman = calculate_boman_index(seq) if seq else 0.0
        heavy_atoms = count_heavy_atoms(seq) if seq else 1
        le = calculate_ligand_efficiency(affinity, heavy_atoms)
        bepr = calculate_per_residue_energy(affinity, len(seq) if seq else 1)

        contacts = analyze_complex_contacts(rec_pdb_path, top_pose_pdb) if top_pose_pdb.exists() else {
            "salt_bridges": 0, "hbonds": 0, "hydrophobic_contacts": 0, "clashes": 0, "key_contacts": []
        }

        composite_score = compute_composite_score(
            affinity_kcal_mol=affinity,
            net_charge=z,
            hydrophobic_moment=mu_h,
            boman_index=boman,
            salt_bridges=contacts["salt_bridges"],
            hbonds=contacts["hbonds"],
            clashes=contacts["clashes"]
        )

        sol_aff = estimate_solution_affinity(
            affinity_kcal_mol=affinity,
            sequence=seq,
            conformation=args.conformation,
            is_cyclic=args.disulfide_lock
        ) if seq else {"entropy_penalty": 0.0, "solution_delta_g": affinity, "solution_kd": "N/A"}

        tox = predict_hemolytic_risk_and_selectivity(
            net_charge=z,
            mean_hydrophobicity=mean_h,
            hydrophobic_moment=mu_h
        )

        summary_data.append({
            "pep_id": pep_id,
            "sequence": seq,
            "length": len(seq),
            "heavy_atoms": heavy_atoms,
            "affinity": affinity,
            "stage1_affinity": affinity,
            "ligand_efficiency": le,
            "bepr": bepr,
            "net_charge_z": z,
            "hydrophobic_moment_muH": mu_h,
            "mean_hydrophobicity": mean_h,
            "boman_index": boman,
            "salt_bridges": contacts["salt_bridges"],
            "hbonds": contacts["hbonds"],
            "hydrophobic_contacts": contacts["hydrophobic_contacts"],
            "clashes": contacts["clashes"],
            "composite_score": composite_score,
            "docked_pdb": top_pose_pdb,
            "complex_pdb": dres.get("complex_pdb"),
            "entropy_penalty": sol_aff["entropy_penalty"],
            "solution_delta_g": sol_aff["solution_delta_g"],
            "solution_kd": sol_aff["solution_kd"],
            "hemolytic_risk": tox["hemolytic_risk"],
            "therapeutic_selectivity": tox["therapeutic_selectivity"]
        })

    # If ensemble mode, group by parent peptide and select optimal conformer per unique peptide
    if args.conformation == "ensemble":
        best_by_parent = {}
        for row in summary_data:
            parent_id = row["pep_id"].split("__conf_")[0]
            conf_name = row["pep_id"].split("__conf_")[1] if "__conf_" in row["pep_id"] else "default"
            row["best_conformation"] = conf_name
            row["parent_pep_id"] = parent_id

            if parent_id not in best_by_parent:
                best_by_parent[parent_id] = row
            else:
                curr_best = best_by_parent[parent_id]
                if (row.get("composite_score", -999), -row.get("affinity", 0)) > \
                   (curr_best.get("composite_score", -999), -curr_best.get("affinity", 0)):
                    best_by_parent[parent_id] = row

        summary_data = list(best_by_parent.values())
        logger.info(f"Ensemble screening filtered {len(summary_data)} optimal conformers from multi-state ensemble.")

    # Sort Stage 1 summary data
    summary_data.sort(key=lambda x: (x.get("composite_score", -999), -x.get("affinity", 0)), reverse=True)
    export_screening_report(
        summary_data=summary_data,
        receptor_pdb=rec_pdb_path,
        out_dir=out_dir,
        top_n=args.top,
        filename="screening_leaderboard_stage1.csv",
        obabel_bin=obabel_bin
    )

    # Stage 2: Induced-Fit Flexible Sidechain Refinement & Rescoring
    refine_n = min(args.refine_top, len(summary_data)) if args.refine_top > 0 else 0
    refined_summary_data = []

    if refine_n > 0:
        print("\n" + "="*70)
        print(f"   STAGE 2: INDUCED-FIT SIDECHAIN REFINEMENT (Top {refine_n} Candidates)")
        print("="*70 + "\n")
        logger.info(f"Refining top {refine_n} candidates with flexible sidechains (exhaustiveness={args.refine_exhaustiveness})...")

        top_stage1_hits = summary_data[:refine_n]
        stage1_docking_dir = out_dir / "docking_poses"
        stage2_docking_dir = out_dir / "docking_poses_stage2"
        ensure_dir(stage2_docking_dir)

        # Prepare ligands from Stage 1 docked poses for local interface refinement
        stage2_ligand_paths = {}
        for hit in top_stage1_hits:
            pep_id = hit["pep_id"]
            stage1_pose_pdb = hit.get("docked_pdb")
            if not stage1_pose_pdb or not Path(stage1_pose_pdb).exists():
                stage1_pose_pdb = stage1_docking_dir / f"{pep_id}_pose1.pdb"
            if Path(stage1_pose_pdb).exists():
                stage2_ligand_paths[pep_id] = Path(stage1_pose_pdb)

        # Run Stage 2 pocket-guided local geometric refinement
        stage2_dock_results = run_parallel_screening(
            ligands=stage2_ligand_paths,
            receptor_pdb=rec_pdb_path,
            grid=grid,
            out_dir=stage2_docking_dir,
            threads=args.threads,
            max_workers=args.max_workers,
            exhaustiveness=args.refine_exhaustiveness,
            num_modes=args.num_modes,
            local_only=True
        )

        for hit in top_stage1_hits:
            pep_id = hit["pep_id"]
            dres = stage2_dock_results.get(pep_id)

            if dres and dres["success"]:
                ref_affinity = dres["best_affinity"]
                ref_pose_pdb = dres.get("docked_pdb")
                if not ref_pose_pdb or not Path(ref_pose_pdb).exists():
                    ref_pose_pdb = stage2_docking_dir / f"{pep_id}_pose1.pdb"

                ref_contacts = analyze_complex_contacts(rec_pdb_path, ref_pose_pdb) if ref_pose_pdb.exists() else {
                    "salt_bridges": hit["salt_bridges"], "hbonds": hit["hbonds"],
                    "hydrophobic_contacts": hit["hydrophobic_contacts"], "clashes": 0, "key_contacts": []
                }

                delta_delta_g = round(ref_affinity - hit["stage1_affinity"], 2)
                ref_le = calculate_ligand_efficiency(ref_affinity, hit["heavy_atoms"])
                ref_bepr = calculate_per_residue_energy(ref_affinity, hit["length"])

                ref_comp = compute_composite_score(
                    affinity_kcal_mol=ref_affinity,
                    net_charge=hit["net_charge_z"],
                    hydrophobic_moment=hit["hydrophobic_moment_muH"],
                    boman_index=hit["boman_index"],
                    salt_bridges=ref_contacts["salt_bridges"],
                    hbonds=ref_contacts["hbonds"],
                    clashes=ref_contacts["clashes"]
                )

                ref_sol_aff = estimate_solution_affinity(
                    affinity_kcal_mol=ref_affinity,
                    sequence=hit.get("sequence", ""),
                    conformation=args.conformation,
                    is_cyclic=args.disulfide_lock
                ) if hit.get("sequence") else {"entropy_penalty": 0.0, "solution_delta_g": ref_affinity, "solution_kd": "N/A"}

                refined_item = dict(hit)
                refined_item.update({
                    "affinity": ref_affinity,
                    "refined_affinity": ref_affinity,
                    "delta_delta_g": delta_delta_g,
                    "ligand_efficiency": ref_le,
                    "bepr": ref_bepr,
                    "salt_bridges": ref_contacts["salt_bridges"],
                    "hbonds": ref_contacts["hbonds"],
                    "hydrophobic_contacts": ref_contacts["hydrophobic_contacts"],
                    "clashes": ref_contacts["clashes"],
                    "composite_score": ref_comp,
                    "docked_pdb": ref_pose_pdb,
                    "complex_pdb": dres.get("complex_pdb"),
                    "solution_delta_g": ref_sol_aff["solution_delta_g"],
                    "solution_kd": ref_sol_aff["solution_kd"],
                    "stage": "Stage 2 (Refined)"
                })
                refined_summary_data.append(refined_item)
            else:
                # Fallback to Stage 1 result
                fallback_item = dict(hit)
                fallback_item["delta_delta_g"] = 0.0
                fallback_item["stage"] = "Stage 1 (Unrefined)"
                refined_summary_data.append(fallback_item)

        # Merge remaining unrefined candidates
        for hit in summary_data[refine_n:]:
            item = dict(hit)
            item["delta_delta_g"] = 0.0
            item["stage"] = "Stage 1 (Screening)"
            refined_summary_data.append(item)

        # Export Refined Report
        csv_path, pml_path = export_screening_report(
            summary_data=refined_summary_data,
            receptor_pdb=rec_pdb_path,
            out_dir=out_dir,
            top_n=refine_n,
            filename="screening_leaderboard_refined.csv",
            obabel_bin=obabel_bin
        )
        export_screening_report(
            summary_data=refined_summary_data,
            receptor_pdb=rec_pdb_path,
            out_dir=out_dir,
            top_n=refine_n,
            filename="screening_leaderboard.csv",
            obabel_bin=obabel_bin
        )
    else:
        logger.info("Stage 2 sidechain refinement skipped (--refine-top 0).")
        refined_summary_data = list(summary_data)
        for hit in refined_summary_data:
            hit["delta_delta_g"] = 0.0
            hit["stage"] = "Stage 1 (Screening)"
        csv_path, pml_path = export_screening_report(
            summary_data=refined_summary_data,
            receptor_pdb=rec_pdb_path,
            out_dir=out_dir,
            top_n=args.top,
            filename="screening_leaderboard.csv",
            obabel_bin=obabel_bin
        )

    # STAGE 3: HADDOCK Physics & Consensus Rescoring
    stage3_n = refine_n if refine_n > 0 else min(args.top, len(refined_summary_data))
    if not getattr(args, "no_ml", False) and stage3_n > 0:
        print("\n" + "="*70)
        print("   STAGE 3: HADDOCK PHYSICS & CONSENSUS RESCORING")
        print("="*70 + "\n")
        logger.info(f"Evaluating top {stage3_n} complexes with HADDOCK explicit physics & consensus scoring...")

        # Determine ML model path (auto-load default pre-trained ensemble if present)
        model_file = getattr(args, "ml_model", None)
        if model_file is None:
            default_pkg_model = Path(__file__).resolve().parent.parent / "data" / "trained_complex_ensemble.joblib"
            if default_pkg_model.exists():
                model_file = default_pkg_model
        ml_scorer = MLPeptideScorer(model_path=model_file)

        complex_dir = out_dir / "top_complexes"
        pocket_list = (
            [r.strip() for r in args.pocket_residues.split(",")]
            if args.pocket_residues
            else grid.get("detected_pocket_residues")
        )
        if pocket_list:
            logger.info(f"Stage 3 AIR Restraints: applying pocket restraints to {len(pocket_list)} target residues.")

        rescore_k = max(1, getattr(args, "rescore_modes", 5))

        for hit in refined_summary_data[:stage3_n]:
            pep_id = hit["pep_id"]
            best_ml_res = None
            best_cpx_pdb = None
            best_mode_idx = 1
            best_consensus = -9999.0

            # Find available docked modes in stage2 or stage1 directories
            search_dirs = [out_dir / "docking_poses_stage2", out_dir / "docking_poses"]
            mode_files = []
            for d in search_dirs:
                if d.exists():
                    m_files = sorted(d.glob(f"{pep_id}_mode*.pdb"))
                    if m_files:
                        mode_files = m_files
                        break

            modes_to_test = min(rescore_k, max(1, len(mode_files))) if mode_files else 1

            for mode_idx in range(1, modes_to_test + 1):
                mode_pose = None
                if mode_files and mode_idx <= len(mode_files):
                    mode_pose = mode_files[mode_idx - 1]
                elif hit.get("docked_pdb") and Path(hit["docked_pdb"]).exists():
                    mode_pose = Path(hit["docked_pdb"])

                mode_cpx = complex_dir / f"complex_{pep_id}_mode{mode_idx}.pdb"
                if mode_pose and mode_pose.exists():
                    create_cocomplex_pdb(rec_pdb_path, mode_pose, mode_cpx)
                elif hit.get("complex_pdb") and Path(hit["complex_pdb"]).exists():
                    mode_cpx = Path(hit["complex_pdb"])
                else:
                    break

                try:
                    cpx_to_eval = Path(mode_cpx)
                    if getattr(args, "relax", False) and mode_idx == 1:
                        from esm_amp_screen.lead_optimizer import relax_complex_interface
                        relaxed_cpx = cpx_to_eval.with_name(f"{cpx_to_eval.stem}_relaxed.pdb")
                        cpx_to_eval = relax_complex_interface(
                            cpx_to_eval,
                            relaxed_cpx,
                            receptor_chain="A",
                            peptide_chain="B",
                            steps=50
                        )

                    ml_res = ml_scorer.score_complex(
                        complex_pdb=cpx_to_eval,
                        hit_meta=hit,
                        pocket_residues=pocket_list
                    )
                    cons = ml_res.get("consensus_score", -9999.0)
                    if cons > best_consensus or best_ml_res is None:
                        best_consensus = cons
                        best_ml_res = ml_res
                        best_cpx_pdb = cpx_to_eval
                        best_mode_idx = mode_idx
                except Exception as e:
                    logger.warning(f"Physics scoring failed for {pep_id} mode {mode_idx}: {e}")

            if best_ml_res is not None:
                hit.update(best_ml_res)
                hit["best_mode"] = best_mode_idx
                if best_cpx_pdb and Path(best_cpx_pdb).exists():
                    final_cpx = complex_dir / f"complex_{pep_id}.pdb"
                    if Path(best_cpx_pdb).resolve() != final_cpx.resolve():
                        import shutil
                        shutil.copy2(best_cpx_pdb, final_cpx)
                    hit["complex_pdb"] = str(final_cpx)
            else:
                hit["best_mode"] = 1

        # Re-export final reports with Consensus and Physics fields included
        if refine_n > 0:
            csv_path, pml_path = export_screening_report(
                summary_data=refined_summary_data,
                receptor_pdb=rec_pdb_path,
                out_dir=out_dir,
                top_n=refine_n,
                filename="screening_leaderboard_refined.csv",
                obabel_bin=obabel_bin
            )
        export_screening_report(
            summary_data=refined_summary_data,
            receptor_pdb=rec_pdb_path,
            out_dir=out_dir,
            top_n=stage3_n,
            filename="screening_leaderboard.csv",
            obabel_bin=obabel_bin
        )

        # Print Stage 3 Consensus Leaderboard
        print("\n" + "="*145)
        print("            STAGE 3: HADDOCK PHYSICS & CONSENSUS LEADERBOARD")
        print("="*145)
        header = f"{'Rank':<5} {'Peptide ID':<16} {'Mode':<6} {'Dock ΔG':<9} {'HADDOCK':<9} {'BSA (Å²)':<9} {'Cons. ΔG':<9} {'Pred Kd':<9} {'Sol. Kd':<10} {'Toxicity':<14} {'Consensus':<10}"
        print(header)
        print("-" * len(header))
        for row in refined_summary_data[:stage3_n]:
            hscore_str = f"{row.get('haddock_score', 0.0):.1f}" if 'haddock_score' in row else "N/A"
            bsa_str = f"{row.get('bsa', 0.0):.1f}" if 'bsa' in row else "N/A"
            ml_dg_str = f"{row.get('ml_delta_g', 0.0):.2f}" if 'ml_delta_g' in row else "N/A"
            pred_kd_str = str(row.get('ml_kd', row.get('estimated_kd', 'N/A')))
            sol_kd_str = str(row.get('solution_kd', 'N/A'))
            tox_str = str(row.get('hemolytic_risk', 'Low'))
            cons_str = f"{row.get('consensus_score', 0.0):.2f}" if 'consensus_score' in row else "N/A"
            mode_str = f"M{row.get('best_mode', 1)}"

            print(
                f"{row.get('rank', 1):<5} {row['pep_id']:<16} {mode_str:<6} {row['affinity']:<9.2f} "
                f"{hscore_str:<9} {bsa_str:<9} {ml_dg_str:<9} "
                f"{pred_kd_str:<9} {sol_kd_str:<10} {tox_str:<14} {cons_str:<10}"
            )
        print("="*145 + "\n")
    elif refine_n > 0:
        # Print Stage 2 Leaderboard if Stage 3 was disabled
        print("\n" + "="*95)
        print("          STAGE 2: INDUCED-FIT REFINED LEADERBOARD (SORTED BY COMPOSITE SCORE)")
        print("="*95)
        header = f"{'Rank':<5} {'Peptide ID':<16} {'Stage 1 ΔG':<12} {'Refined ΔG':<12} {'ΔΔG Gain':<10} {'LE':<8} {'BEPR':<8} {'Salt Br.':<10} {'Composite':<10}"
        print(header)
        print("-" * len(header))
        for row in refined_summary_data[:refine_n]:
            print(
                f"{row.get('rank', 1):<5} {row['pep_id']:<16} {row['stage1_affinity']:<12.2f} "
                f"{row['affinity']:<12.2f} {row.get('delta_delta_g', 0.0):<10.2f} "
                f"{row['ligand_efficiency']:<8.3f} {row['bepr']:<8.2f} "
                f"{row['salt_bridges']:<10} {row['composite_score']:<10.2f}"
            )
        print("="*95 + "\n")
    else:
        print("\n" + "="*85)
        print("                    TOP SCREENING CANDIDATES (RANKED BY COMPOSITE SCORE)")
        print("="*85)
        header = f"{'Rank':<5} {'Peptide ID':<16} {'Affinity (kcal/mol)':<20} {'Charge (z)':<12} {'LE':<8} {'Salt Br.':<10} {'Composite':<10}"
        print(header)
        print("-" * len(header))
        for row in summary_data[:args.top]:
            print(
                f"{row['rank']:<5} {row['pep_id']:<16} {row['affinity']:<20.2f} "
                f"{row['net_charge_z']:<12.2f} {row['ligand_efficiency']:<8.3f} "
                f"{row['salt_bridges']:<10} {row['composite_score']:<10.2f}"
            )
        print("="*85 + "\n")

    logger.info(f"Results summary: {csv_path}")
    logger.info(f"PyMOL session:   {pml_path}")
    logger.info(f"HTML 3D Report:  {out_dir / 'screening_report.html'}")
    logger.info(f"Top complex PDBs: {out_dir / 'top_complexes'}")


def main():
    parser = argparse.ArgumentParser(
        prog="esm-amp-screen",
        description="High-Throughput ESM-Powered Protein-Peptide Virtual Screening, 7-Ray PSP Multi-Pocket Blind Docking, Physics Engine, & In Silico Lead Optimizer"
    )
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Command: run / screen / target (all-in-one target-directed virtual screening)
    p_run = subparsers.add_parser("run", aliases=["screen", "target"], help="Run target-directed protein-peptide virtual screening pipeline (multi-mode docking + HADDOCK physics)")
    p_run.add_argument("--receptor", "-r", required=True, help="Receptor PDB file path or 4-letter RCSB ID (e.g. 2mlt)")
    p_run.add_argument("--rec-chain", default="A", help="Receptor chain ID to dock against (default: A). Extraneous co-crystal chains are safely removed.")
    p_run.add_argument("--library", "-l", required=True, help="Peptide library FASTA file, or folder of PDB structures")
    p_run.add_argument("--output", "-o", default="./screening_results", help="Output directory")
    p_run.add_argument("--center", nargs=3, type=float, help="Grid box center (x y z in Å). Auto-calculated if omitted.")
    p_run.add_argument("--size", nargs=3, type=float, help="Grid box dimensions (sx sy sz in Å). Auto-calculated if omitted.")
    p_run.add_argument("--pocket-residues", help="Comma-separated pocket residues (e.g. 'A:45,A:48,A:52')")
    p_run.add_argument("--auto-pocket", dest="auto_pocket", action="store_true", default=True,
                       help="Automatically detect druggable surface pockets using geometric concavity grid if --center is omitted (default: True)")
    p_run.add_argument("--no-auto-pocket", dest="auto_pocket", action="store_false",
                       help="Disable automatic pocket detection and use whole-protein search box")
    p_run.add_argument("--disulfide-lock", dest="disulfide_lock", action="store_true", default=True,
                       help="Perceive and lock intra-peptide disulfide bonds (SG-SG) and cyclic topology during docking (default: True)")
    p_run.add_argument("--no-disulfide-lock", dest="disulfide_lock", action="store_false",
                       help="Disable disulfide locking and treat cysteines as standard flexible thiols")
    p_run.add_argument("--padding", type=float, default=8.0, help="Grid box padding around receptor or pocket (Å)")
    p_run.add_argument("--conformation", "-c", choices=["auto", "helix", "sheet", "extended", "ppii", "ensemble"], default="auto",
                       help="Secondary structure prior conformation for AMPs: 'auto' (sequence-based propensity detection, default), 'helix', 'sheet', 'extended', 'ppii', or 'ensemble'")
    p_run.add_argument("--flexibility", choices=["rigid", "sidechain"], default="rigid",
                       help="Peptide flexibility mode: 'rigid' (geometric surface search, default)")
    p_run.add_argument("--threads", "-t", type=int, default=os.cpu_count() or 4,
                       help="Total CPU worker threads to allocate across docking processes (default: all available CPUs)")
    p_run.add_argument("--max-workers", type=int, default=None,
                       help="Maximum concurrent docking worker processes (default: min(threads, 4) to prevent RAM thrashing)")
    p_run.add_argument("--exhaustiveness", "-e", type=int, default=2,
                       help="Stage 1 geometric search exhaustiveness / rotational density (default: 2)")
    p_run.add_argument("--refine-top", type=int, default=3,
                       help="Stage 2: Number of top candidates to refine with local simplex minimization (default: 3, set 0 to disable)")
    p_run.add_argument("--refine-exhaustiveness", type=int, default=4,
                       help="Stage 2: Geometric search exhaustiveness for local interface refinement (default: 4)")
    p_run.add_argument("--num-modes", type=int, default=9, help="Max binding modes per peptide (default: 9)")
    p_run.add_argument("--top", type=int, default=5, help="Number of top hits to export complex PDBs for (default: 5)")
    p_run.add_argument("--rescore-modes", type=int, default=5,
                       help="Number of docked binding modes to evaluate and rescore per top candidate in Stage 3 (default: 5)")
    p_run.add_argument("--hf-token", help="HuggingFace API token for ESMFold inference endpoint")
    p_run.add_argument("--obabel-bin", help="Path to OpenBabel binary (default: search PATH)")
    p_run.add_argument("--ml-model", help="Path to custom pre-trained ML model file (.pkl / .joblib)")
    p_run.add_argument("--relax", action="store_true", default=False,
                       help="Perform interface force-field relaxation (OpenBabel UFF) on top poses before Stage 3 scoring")
    p_run.add_argument("--no-ml", action="store_true", help="Disable Stage 3 HADDOCK physics and ML scoring")
    p_run.add_argument("--blind-pockets", type=int, default=1,
                       help="Number of top druggable surface pockets to dock against in blind docking mode (default: 1; set to 5 for comprehensive 5-pocket blind docking)")
    p_run.add_argument("--blind-docking", action="store_true", default=False,
                       help="Enable full multi-pocket blind docking across the top 5 predicted receptor surface pockets")
    p_run.set_defaults(func=run_pipeline)

    # Command: pockets (standalone cavity detection and druggability scoring)
    p_pock = subparsers.add_parser("pockets", help="Detect, profile, and score the top druggable binding pockets on a receptor PDB")
    p_pock.add_argument("--receptor", "-r", required=True, help="Receptor PDB file path or 4-letter RCSB ID (e.g. 1ycr)")
    p_pock.add_argument("--rec-chain", default="A", help="Receptor chain ID (default: A)")
    p_pock.add_argument("--top-pockets", "-k", type=int, default=5, help="Number of top druggable pockets to detect (default: 5)")
    p_pock.add_argument("--output", "-o", default="./pocket_analysis", help="Output directory for pocket reports and PyMOL scripts")
    p_pock.add_argument("--padding", type=float, default=8.0, help="Grid box padding around pocket (Å)")
    p_pock.add_argument("--min-distance", type=float, default=8.0, help="Minimum spatial distance between detected pocket centers in Å (default: 8.0)")
    p_pock.set_defaults(func=run_pockets_cli)

    # Command: membrane (sequence-first membrane biophysical profiling with ESM-2 & Wimley-White biophysics)
    p_mem = subparsers.add_parser("membrane", help="Sequence-first membrane biophysical profiling using ESM-2 representations & Wimley-White thermodynamics (no docking required)")
    p_mem.add_argument("--library", "-l", required=True, help="Peptide library FASTA file, or CSV containing a 'sequence' column")
    p_mem.add_argument("--output", "-o", default="./membrane_screening_results", help="Output directory for membrane screening leaderboard")
    p_mem.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="ESM-2 model name or checkpoint path (default: facebook/esm2_t12_35M_UR50D)")
    p_mem.add_argument("--device", default="cpu", help="Device to run ESM-2 ('cpu' or 'cuda', default: cpu)")
    p_mem.add_argument("--top", type=int, default=20, help="Number of top hits to display in CLI leaderboard (default: 20)")
    p_mem.add_argument("--batch-size", type=int, default=32, help="ESM-2 embedding batch size (default: 32)")
    p_mem.add_argument("--no-esm", action="store_true", default=False, help="Disable ESM-2 and use pure Wimley-White biophysics")
    p_mem.set_defaults(func=run_membrane_screening_cli)

    # Command: optimize-hit (in silico alanine scan & lead optimization)
    p_opt = subparsers.add_parser("optimize-hit", help="In silico lead optimization, alanine scanning, and affinity maturation")
    p_opt.add_argument("--complex", "-c", required=True, help="Input docked complex PDB file (co-complex with receptor and peptide chains)")
    p_opt.add_argument("--rec-chain", default="A", help="Receptor chain ID in complex (default: A)")
    p_opt.add_argument("--pep-chain", default="B", help="Peptide chain ID in complex (default: B)")
    p_opt.add_argument("--output", "-o", help="Output directory for optimization results (default: <complex_dir>/lead_optimization)")
    p_opt.add_argument("--mode", choices=["all", "alanine", "rational"], default="all",
                       help="Optimization mode: 'alanine' (hotspot scan), 'rational' (target mutations), or 'all' (default: all)")
    p_opt.add_argument("--positions", help="Comma-separated 1-based peptide residue numbers to mutate (e.g. '1,4,7'). If omitted, scans tolerant interface positions.")
    p_opt.add_argument("--substitutions", help="Comma-separated candidate amino acids (e.g. 'W,F,Y,R,K,E,D'). Default: rational library.")
    p_opt.add_argument("--pocket-residues", help="Comma-separated target pocket residues (e.g. 'A:45,A:48') for AIR evaluation")
    p_opt.add_argument("--top-n", type=int, default=10, help="Number of top variants to report (default: 10)")
    p_opt.add_argument("--threads", "-t", type=int, default=os.cpu_count() or 4,
                       help="Number of CPU worker threads for parallel mutation scanning (default: all available CPUs)")
    p_opt.set_defaults(func=run_lead_optimization_cli)

    # Command: score (standalone complex rescoring)
    p_score = subparsers.add_parser("score", help="Compute HADDOCK physics, solution thermodynamics, and ML scores for an existing complex")
    p_score.add_argument("--complex", "-c", required=True, help="Input co-complex PDB file (receptor + peptide)")
    p_score.add_argument("--rec-chain", default="A", help="Receptor chain ID (default: A)")
    p_score.add_argument("--pep-chain", default="B", help="Peptide chain ID (default: B)")
    p_score.add_argument("--pocket-residues", help="Comma-separated target pocket residues (e.g. 'A:45,A:48') for AIR evaluation")
    p_score.add_argument("--relax", action="store_true", default=False, help="Perform interface force-field relaxation before scoring")
    p_score.add_argument("--ml-model", help="Path to custom pre-trained ML model file (.pkl / .joblib)")
    p_score.add_argument("--output", "-o", help="Optional output directory to save relaxed complex and scoring report")
    p_score.set_defaults(func=run_score_cli)

    # Command: benchmark (compare against crystal structure or run benchmark suite)
    p_bench = subparsers.add_parser("benchmark", help="Compare docked pose against experimental crystal structure, or run benchmark suite")
    p_bench.add_argument("--mode", choices=["target"], default="target",
                         help="Benchmark mode: 'target' (co-crystal docking recovery across 14 complexes, default: target)")
    p_bench.add_argument("--complex", "-c", help="Predicted complex PDB (target + peptide)")
    p_bench.add_argument("--reference", "-r", help="Reference co-crystal PDB")
    p_bench.add_argument("--run-all", action="store_true", default=False, help="Run standardized validation suite across all 14 benchmark co-crystals")
    p_bench.add_argument("--benchmarks-dir", default="./benchmarks", help="Directory of benchmark PDB files (default: ./benchmarks)")
    p_bench.add_argument("--rec-chain", default="A", help="Receptor chain in predicted complex (default: A)")
    p_bench.add_argument("--pep-chain", default="B", help="Peptide chain in predicted complex (default: B)")
    p_bench.add_argument("--ref-rec-chain", help="Receptor chain in reference PDB (default: same as --rec-chain)")
    p_bench.add_argument("--ref-pep-chain", help="Peptide chain in reference PDB (default: same as --pep-chain)")
    p_bench.add_argument("--out-superimposed", "-o", help="Path to save receptor-superimposed complex PDB")
    p_bench.set_defaults(func=run_benchmark_cli)

    args = parser.parse_args()
    args.func(args)


def run_pockets_cli(args: argparse.Namespace) -> None:
    """CLI runner to detect, profile, and score the top druggable binding pockets on a receptor."""
    import pandas as pd
    from esm_amp_screen.receptor_prep import clean_receptor_pdb, detect_surface_cavities
    from esm_amp_screen.esm_folder import fetch_rcsb_pdb
    from esm_amp_screen.reporter import generate_pockets_pymol_script

    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)

    print("\n" + "="*110)
    print("      ESM-AMP-SCREEN: DRUGGABLE SURFACE POCKET DETECTION & GEOMETRIC CAVITY PROFILING")
    print("      High-Resolution 7-Ray PSP LIGSITE Grid Concavity & 0-100 Druggability Scoring")
    print("="*110 + "\n")

    receptor_arg = args.receptor
    rec_chain = getattr(args, "rec_chain", "A")
    rec_pdb_path = out_dir / "target_receptor.pdb"
    if Path(receptor_arg).exists():
        rec_pdb_path = clean_receptor_pdb(Path(receptor_arg), rec_pdb_path, receptor_chain=rec_chain)
    elif len(receptor_arg) == 4 and receptor_arg.isalnum():
        fetch_rcsb_pdb(receptor_arg, rec_pdb_path)
        rec_pdb_path = clean_receptor_pdb(rec_pdb_path, out_dir / "target_receptor_clean.pdb", receptor_chain=rec_chain)
    else:
        raise FileNotFoundError(f"Receptor file not found and not a valid 4-letter PDB ID: {receptor_arg}")

    top_k = getattr(args, "top_pockets", 5)
    padding = getattr(args, "padding", 8.0)
    min_dist = getattr(args, "min_distance", 8.0)

    pockets = detect_surface_cavities(
        rec_pdb_path,
        max_pockets=top_k,
        padding=padding,
        min_pocket_distance=min_dist
    )

    if not pockets:
        logger.error(f"No surface cavities detected on receptor {rec_pdb_path.name}.")
        return

    print(f"Receptor File:     {rec_pdb_path.name} (Chain: {rec_chain})")
    print(f"Pockets Detected:  {len(pockets)} distinct druggable binding sites identified\n")

    header = f"{'ID':<4} {'Druggability':<14} {'Class':<16} {'Volume (Å³)':<13} {'Center (x, y, z)':<26} {'Enclosure':<11} {'Hydro%':<8} {'Lining Res':<10}"
    print(header)
    print("-" * len(header))
    for p in pockets:
        p_id = f"P#{p['pocket_id']}"
        score_str = f"{p['druggability_score']:.1f} / 100"
        d_class = p['druggability_class'].split(' (')[0]
        vol_str = f"{p['volume_angstrom3']:.1f}"
        center_str = f"({p['center_x']:>6.1f}, {p['center_y']:>6.1f}, {p['center_z']:>6.1f})"
        enc_str = f"{p['enclosure_score']:.2f} / 7"
        hydro_str = f"{p['hydrophobic_ratio']*100:.0f}%"
        res_str = f"{len(p['contact_residues'])} residues"
        print(f"{p_id:<4} {score_str:<14} {d_class:<16} {vol_str:<13} {center_str:<26} {enc_str:<11} {hydro_str:<8} {res_str:<10}")
    print("-" * len(header))

    # Detailed Pocket Breakdown
    print("\nDetailed Lining Hotspots & Dimensions:")
    for p in pockets:
        hotspots = ", ".join(p['contact_residues_formatted'][:8])
        if len(p['contact_residues_formatted']) > 8:
            hotspots += f", ... (+{len(p['contact_residues_formatted']) - 8} more)"
        print(f"  • Pocket #{p['pocket_id']}: [{p['druggability_class']}]")
        print(f"    - Grid Box Size:  ({p['size_x']} x {p['size_y']} x {p['size_z']}) Å")
        print(f"    - Probe Points:   {p['num_points']} surface points ({p['volume_angstrom3']} Å³)")
        print(f"    - Lining Hotspots:{hotspots}")

    # Export CSV
    df = pd.DataFrame(pockets)
    csv_out = out_dir / "pockets_summary.csv"
    df.to_csv(csv_out, index=False)

    # Export PyMOL script
    pml_out = out_dir / "view_pockets.pml"
    generate_pockets_pymol_script(rec_pdb_path, pockets, pml_out)

    print("\n" + "="*110)
    print(f"[OK] Pocket summary saved to:         {csv_out}")
    print(f"[OK] Multi-color PyMOL visualizer to: {pml_out}")
    print("="*110 + "\n")


def run_benchmark_cli(args: argparse.Namespace) -> None:
    if getattr(args, "run_all", False) or (not args.complex and not args.reference):
        from run_benchmarks import run_all_benchmarks
        b_dir = Path(args.benchmarks_dir).resolve()
        o_csv = b_dir / "benchmark_results.csv"
        run_all_benchmarks(b_dir, o_csv)
        return

    if not args.complex or not args.reference:
        logger.error("Both --complex and --reference must be specified, or use --run-all to run the benchmark suite.")
        sys.exit(1)

    from esm_amp_screen.benchmark import evaluate_docked_complex
    complex_pdb = Path(args.complex).resolve()
    ref_pdb = Path(args.reference).resolve()
    out_super = Path(args.out_superimposed).resolve() if args.out_superimposed else None

    print("\n" + "="*80)
    print("      CAPRI BENCHMARK & COMPARISON EVALUATION")
    print("="*80 + "\n")

    metrics = evaluate_docked_complex(
        docked_pdb=complex_pdb,
        reference_pdb=ref_pdb,
        docked_rec_chain=args.rec_chain,
        docked_pep_chain=args.pep_chain,
        ref_rec_chain=args.ref_rec_chain or args.rec_chain,
        ref_pep_chain=args.ref_pep_chain or args.pep_chain,
        out_superimposed_pdb=out_super
    )

    print(f"Target Complex:       {complex_pdb.name}")
    print(f"Reference Crystal:    {ref_pdb.name}")
    print("-" * 80)
    print(f"Receptor Cα RMSD:     {metrics['receptor_ca_rmsd']:.3f} Å")
    print(f"Ligand Backbone RMSD: {metrics['l_rmsd_backbone']:.3f} Å  (L-RMSD_bb)")
    print(f"Ligand All-Atom RMSD: {metrics['l_rmsd_all_atom']:.3f} Å  (L-RMSD_all)")
    print(f"Interface RMSD:       {metrics['i_rmsd']:.3f} Å  (I-RMSD)")
    print(f"Native Contacts:      {metrics['f_nat_percent']:.1f}% recovered ({metrics['native_contacts_count']} total)")
    print(f"CAPRI Classification: {metrics['capri_quality']}")
    print("="*80 + "\n")


def run_lead_optimization_cli(args: argparse.Namespace) -> None:
    """CLI runner for in silico alanine scanning and lead optimization."""
    import pandas as pd
    from esm_amp_screen.lead_optimizer import LeadOptimizer

    complex_pdb = Path(args.complex).resolve()
    out_dir = Path(args.output).resolve() if args.output else complex_pdb.parent / "lead_optimization"
    ensure_dir(out_dir)

    print("\n" + "="*95)
    print("        IN SILICO LEAD OPTIMIZATION & MUTATION SCANNING PIPELINE")
    print("="*95 + "\n")

    pocket_list = [r.strip() for r in args.pocket_residues.split(",")] if args.pocket_residues else None

    optimizer = LeadOptimizer(
        complex_pdb=complex_pdb,
        receptor_chain=args.rec_chain,
        peptide_chain=args.pep_chain,
        out_dir=out_dir,
        pocket_residues=pocket_list,
        threads=args.threads
    )

    target_positions = None
    if args.positions:
        target_positions = [int(p.strip()) for p in args.positions.split(",") if p.strip()]

    target_substitutions = None
    if args.substitutions:
        target_substitutions = [s.strip().upper() for s in args.substitutions.split(",") if s.strip()]

    # 1. Alanine Scan
    if args.mode in ("all", "alanine"):
        print("\n" + "="*85)
        print("                 IN SILICO ALANINE SCAN: HOTSPOT IDENTIFICATION")
        print("="*85)
        ala_results = optimizer.run_alanine_scan(threads=args.threads)
        if ala_results:
            df_ala = pd.DataFrame(ala_results)
            csv_ala = out_dir / "alanine_scan_report.csv"
            df_ala.to_csv(csv_ala, index=False)

            header = f"{'Pos':<6} {'Residue':<10} {'Mutant':<10} {'ΔHADDOCK':<12} {'ΔΔG (kcal/mol)':<16} {'BSA (Å²)':<12} {'Classification':<25}"
            print(header)
            print("-" * len(header))
            for r in ala_results:
                dh_str = f"{r['delta_haddock']:+.2f}"
                ddg_str = f"{r['delta_delta_g']:+.2f}"
                print(
                    f"{r['position']:<6} {r['wt_res']:<10} {r['mut_id']:<10} "
                    f"{dh_str:<12} {ddg_str:<16} "
                    f"{r['bsa']:<12.1f} {r['classification']:<25}"
                )
            print("="*85)
            logger.info(f"Saved alanine scan report to: {csv_ala}\n")

    # 2. Lead Optimization Mutations
    if args.mode in ("all", "rational"):
        print("\n" + "="*125)
        print("         LEAD OPTIMIZATION: AFFINITY MATURATION LEADERBOARD (TARGET: -0.8 to -1.5 kcal/mol)")
        print("="*125)
        variants = optimizer.run_lead_optimization(
            candidate_positions=target_positions,
            candidate_amino_acids=target_substitutions,
            top_n=args.top_n,
            threads=args.threads
        )

        if variants:
            df_mut = pd.DataFrame(variants)
            csv_mut = out_dir / "mutation_scan_leaderboard.csv"
            df_mut.to_csv(csv_mut, index=False)

            header = f"{'Rank':<5} {'Mutation':<10} {'ΔHADDOCK':<10} {'ΔΔG (kcal/mol)':<16} {'Kd Fold Boost':<15} {'Sol. Kd':<12} {'Toxicity':<12} {'Biophysical Mechanism':<35}"
            print(header)
            print("-" * len(header))
            for i, v in enumerate(variants, 1):
                boost_str = f"{v['kd_fold_improvement']:.1f}x" if v['delta_delta_g'] < 0 else f"{v['kd_fold_improvement']:.2f}x"
                dh_str = f"{v['delta_haddock']:+.1f}"
                ddg_str = f"{v['delta_delta_g']:+.2f}"
                print(
                    f"{i:<5} {v['mutation']:<10} {dh_str:<10} "
                    f"{ddg_str:<16} {boost_str:<15} "
                    f"{v['solution_kd']:<12} {v['hemolytic_risk']:<12} {v['mechanism'][:35]:<35}"
                )
            print("="*125)
            logger.info(f"Saved mutation leaderboard to: {csv_mut}")
            logger.info(f"Optimized co-complex PDB structures saved to: {out_dir / 'optimized_variants'}\n")


def run_score_cli(args: argparse.Namespace) -> None:
    """CLI runner to score an existing protein-peptide complex."""
    import gemmi
    from esm_amp_screen.physics_engine import calculate_haddock_physics_score
    from esm_amp_screen.amp_analyzer import (
        analyze_complex_contacts,
        estimate_solution_affinity
    )
    from esm_amp_screen.ml_scorer import MLPeptideScorer
    from esm_amp_screen.lead_optimizer import relax_complex_interface, AA_3TO1

    complex_pdb = Path(args.complex).resolve()
    if not complex_pdb.exists():
        raise FileNotFoundError(f"Complex PDB not found: {complex_pdb}")

    print("\n" + "="*95)
    print("      STANDALONE COMPLEX SCORING & SOLUTION THERMODYNAMICS BREAKDOWN")
    print("="*95 + "\n")

    eval_pdb = complex_pdb
    if getattr(args, "relax", False):
        out_rel = complex_pdb.with_name(f"{complex_pdb.stem}_relaxed.pdb")
        eval_pdb = relax_complex_interface(
            complex_pdb,
            out_rel,
            receptor_chain=args.rec_chain,
            peptide_chain=args.pep_chain,
            steps=50
        )
        print(f"Interface relaxed: {eval_pdb.name}\n")

    pocket_list = [r.strip() for r in args.pocket_residues.split(",")] if args.pocket_residues else None

    # Parse peptide sequence
    st = gemmi.read_structure(str(eval_pdb))
    pep_chain = None
    for c in st[0]:
        if c.name == args.pep_chain:
            pep_chain = c
            break
    pep_seq = ""
    if pep_chain:
        pep_seq = "".join(AA_3TO1.get(r.name.upper(), "X") for r in pep_chain)

    phys = calculate_haddock_physics_score(
        eval_pdb,
        pocket_residues=pocket_list,
        receptor_chain=args.rec_chain,
        peptide_chain=args.pep_chain
    )
    contacts = analyze_complex_contacts(
        eval_pdb,
        eval_pdb,
        receptor_chain=args.rec_chain,
        peptide_chain=args.pep_chain
    )

    sol_aff = estimate_solution_affinity(
        affinity_kcal_mol=phys.get("delta_g", -8.0),
        sequence=pep_seq
    )

    model_file = getattr(args, "ml_model", None)
    if model_file is None:
        default_pkg_model = Path(__file__).resolve().parent.parent / "data" / "trained_complex_ensemble.joblib"
        if default_pkg_model.exists():
            model_file = default_pkg_model

    ml_scorer = MLPeptideScorer(model_path=model_file)
    ml_res = ml_scorer.score_complex(
        complex_pdb=eval_pdb,
        hit_meta={"pep_id": complex_pdb.stem, "sequence": pep_seq},
        pocket_residues=pocket_list
    )

    print(f"Complex File:        {complex_pdb.name}")
    print(f"Receptor Chain:      {args.rec_chain}")
    print(f"Peptide Chain:       {args.pep_chain} (Sequence: {pep_seq}, Length: {len(pep_seq)} aa)")
    print("-" * 95)
    print("1. HADDOCK 2.4 EXPLICIT PHYSICS BREAKDOWN:")
    print(f"   • HADDOCK Score:     {phys['haddock_score']:.2f}  (1.0*Evdw + 0.2*Eelec + 1.0*Edesolv + 0.1*Eair)")
    print(f"   • Van der Waals:     {phys['e_vdw']:.2f} kcal/mol")
    print(f"   • Electrostatics:    {phys['e_elec']:.2f} kcal/mol (Distance-dependent dielectric ε(r)=4r)")
    print(f"   • Desolvation:       {phys['e_desolv']:.2f} kcal/mol (Eisenberg & McLachlan atomic ASP model)")
    print(f"   • AIR Restraints:    {phys['e_air']:.2f} kcal/mol (Active pocket surface contacts)")
    print(f"   • Buried Surface:    {phys['bsa']:.1f} Å² (BSA = SASA_rec + SASA_pep - SASA_cpx)")
    print("-" * 95)
    print("2. SOLUTION THERMODYNAMICS & EXPERIMENTAL Kd:")
    print(f"   • Raw Grid ΔG:       {phys.get('delta_g', -8.0):.2f} kcal/mol")
    print(f"   • Entropy Penalty:   +{sol_aff['entropy_penalty']:.2f} kcal/mol (Conformational freezing & cratic association)")
    print(f"   • Solution ΔG (sol): {sol_aff['solution_delta_g']:.2f} kcal/mol")
    print(f"   • Predicted Kd (sol):{sol_aff['solution_kd']}")
    print("-" * 95)
    print("3. INTERMOLECULAR CONTACTS:")
    print(f"   • Hydrogen Bonds:    {contacts['hbonds']}")
    print(f"   • Salt Bridges:      {contacts['salt_bridges']}")
    print(f"   • Hydrophobic Cont.: {contacts['hydrophobic_contacts']}")
    print(f"   • Clashes:           {contacts['clashes']}")
    print("-" * 95)
    print("4. MACHINE LEARNING & CONSENSUS:")
    print(f"   • ML Predicted pKd:  {ml_res.get('ml_pkd', 'N/A')}")
    print(f"   • ML Predicted ΔG:   {ml_res.get('ml_delta_g', 'N/A')} kcal/mol")
    print(f"   • ML Kd:             {ml_res.get('ml_kd', 'N/A')}")
    print(f"   • Consensus Score:   {ml_res.get('consensus_score', 'N/A'):.2f}")
    print("="*95 + "\n")


def run_membrane_screening_cli(args: argparse.Namespace) -> None:
    """CLI runner for direct sequence-first membrane AMP screening."""
    import pandas as pd
    from esm_amp_screen.membrane_scorer import evaluate_membrane_activity
    from esm_amp_screen.esm_embedder import ESM2Embedder

    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)

    print("\n" + "="*115)
    print("      ESM-AMP-SCREEN: MEMBRANE BIOPHYSICAL PROFILING & PERMEABILIZATION SCREENING")
    print("      Sequence Biophysics: ESM-2 Neural Representations + Wimley-White Interfacial Thermodynamics")
    print("="*115 + "\n")

    # 1. Parse Input Sequences
    lib_path = Path(args.library)
    sequences = {}
    if lib_path.is_file():
        if lib_path.suffix.lower() in (".fasta", ".fa", ".faa"):
            sequences = read_fasta(lib_path)
        elif lib_path.suffix.lower() in (".csv", ".tsv"):
            sep = "," if lib_path.suffix.lower() == ".csv" else "\t"
            df_in = pd.read_csv(lib_path, sep=sep)
            seq_col = next((c for c in df_in.columns if c.lower() in ("sequence", "seq", "peptide")), None)
            id_col = next((c for c in df_in.columns if c.lower() in ("id", "name", "pep_id")), None)
            if not seq_col:
                raise ValueError(f"CSV file must have a 'sequence' or 'seq' column. Found: {list(df_in.columns)}")
            for idx, r in df_in.iterrows():
                pep_id = str(r[id_col]) if id_col else f"pep_{idx+1}"
                sequences[pep_id] = str(r[seq_col]).strip()
        else:
            with open(lib_path) as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        sequences[f"pep_{idx}"] = line
    else:
        lib_str = str(args.library).strip().upper()
        if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in lib_str) and len(lib_str) >= 4:
            sequences = {"input_seq": lib_str}
        else:
            raise FileNotFoundError(f"Peptide library file not found: {lib_path}")

    if not sequences:
        raise ValueError(f"No valid peptide sequences found in {lib_path}")

    logger.info(f"Loaded {len(sequences)} peptide sequences for membrane screening.")

    # 2. ESM-2 Sequence Model (Tier 0)
    embedder = None
    if not getattr(args, "no_esm", False):
        model_name = getattr(args, "model", "facebook/esm2_t12_35M_UR50D")
        device = getattr(args, "device", "cpu")
        embedder = ESM2Embedder(model_name=model_name, device=device)
        logger.info(f"Using ESM-2 model: {model_name} on {device}")

    # 3. Sequence Evaluation
    records = []
    logger.info("Computing Wimley-White interfacial partitioning and biophysical parameters...")

    for pep_id, seq in sequences.items():
        try:
            mem = evaluate_membrane_activity(seq)
        except Exception as e:
            logger.warning(f"Skipping {pep_id}: {e}")
            continue

        records.append({
            "id": pep_id,
            "sequence": seq,
            "length": mem["length"],
            "mpi_score": mem["membrane_permeabilization_index"],
            "net_charge_z": mem["net_charge_z"],
            "charge_density": mem["charge_density_z_per_res"],
            "dg_interfacial": mem["mean_dg_interfacial_kcal"],
            "dg_octanol": mem["total_dg_octanol_kcal"],
            "mu_h_alpha": mem["mu_h_alpha_100deg"],
            "mu_h_beta": mem["mu_h_beta_160deg"],
            "favored_conformation": mem["favored_conformation"],
            "hemolysis_risk": mem["hemolysis_risk_probability"],
            "hemolysis_tier": mem["hemolysis_tier"],
            "therapeutic_index": mem["therapeutic_index"],
            "verdict": mem["verdict"]
        })

    if not records:
        logger.error("No valid sequences could be scored.")
        return

    # 4. Sort and assign percentile ranks by Membrane Permeabilization Index (MPI)
    records.sort(key=lambda x: x["mpi_score"], reverse=True)
    n_rec = len(records)
    for i, r in enumerate(records, 1):
        r["rank"] = i
        r["percentile"] = round(100.0 * (1.0 - (i - 1) / max(1, n_rec)), 1)

    df_out = pd.DataFrame(records)
    csv_out = out_dir / "membrane_biophysics_leaderboard.csv"
    df_out.to_csv(csv_out, index=False)

    # 5. CLI Printout
    top_n = min(getattr(args, "top", 20), n_rec)
    header = (
        f"{'Rank':<5} {'ID':<18} {'Len':<5} "
        f"{'MPI':<7} {'Charge':<8} {'ΔG_if':<8} {'μH(α)':<8} {'Hemolysis':<16} {'TI':<6} {'Triage Verdict':<32}"
    )
    print(header)
    print("-" * len(header))
    for r in records[:top_n]:
        print(
            f"{r['rank']:<5} {r['id'][:16]:<18} {r['length']:<5} "
            f"{r['mpi_score']:<7.1f} {r['net_charge_z']:<8.1f} "
            f"{r['dg_interfacial']:<8.2f} {r['mu_h_alpha']:<8.2f} {r['hemolysis_tier'][:14]:<16} "
            f"{r['therapeutic_index']:<6.1f} {r['verdict'][:32]:<32}"
        )
    print("-" * len(header))

    t1_count = sum(1 for r in records if "Tier 1" in r["verdict"])
    t2_count = sum(1 for r in records if "Tier 2" in r["verdict"])
    t3_count = sum(1 for r in records if "Tier 3" in r["verdict"])
    t4_count = sum(1 for r in records if "Tier 4" in r["verdict"])

    print("\nBiophysical Profiling Summary Across Library:")
    print(f"  • Tier 1 (High Potency & Selective Permeabilizer): {t1_count} ({t1_count/n_rec*100:.1f}%)")
    print(f"  • Tier 2 (Moderate Activity):                     {t2_count} ({t2_count/n_rec*100:.1f}%)")
    print(f"  • Tier 3 (Weak / Marginal Activity):              {t3_count} ({t3_count/n_rec*100:.1f}%)")
    print(f"  • Tier 4 (Inactive / Non-Membrane Active):        {t4_count} ({t4_count/n_rec*100:.1f}%)")
    print(f"\n[OK] Complete results exported to: {csv_out}\n")


if __name__ == "__main__":
    main()

