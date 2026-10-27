"""
Reporting & Visualization Module.
Exports ranked CSV leaderboard, target-peptide co-complex PDB structures,
and interactive PyMOL visualization scripts.
"""

import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import gemmi

from esm_amp_screen.utils import logger, ensure_dir
from esm_amp_screen.peptide_prep import create_docked_pristine_pdb
from esm_amp_screen.html_reporter import generate_interactive_html_report


def estimate_kd(affinity_kcal_mol: float, temp_kelvin: float = 298.15) -> str:
    """
    Estimates dissociation constant (Kd) from Gibbs free energy:
    ΔG = R * T * ln(Kd)  =>  Kd = exp(ΔG / (R * T))
    R = 1.9872e-3 kcal/(mol*K)
    """
    if affinity_kcal_mol >= 0:
        return "> 1.0 mM"
    
    r = 1.9872e-3
    kd_molar = math.exp(affinity_kcal_mol / (r * temp_kelvin))

    if kd_molar < 1e-9:
        return f"{kd_molar * 1e12:.1f} pM"
    elif kd_molar < 1e-6:
        return f"{kd_molar * 1e9:.1f} nM"
    elif kd_molar < 1e-3:
        return f"{kd_molar * 1e6:.1f} µM"
    else:
        return f"{kd_molar * 1e3:.1f} mM"


def convert_docked_pdbqt_to_pdb(
    docked_pdbqt: Path,
    out_pdb: Path,
    obabel_bin: str = "obabel"
) -> Optional[Path]:
    """Converts the top pose (Model 1) of a docked PDBQT file to PDB using OpenBabel."""
    out_pdb = Path(out_pdb)
    docked_pdbqt = Path(docked_pdbqt)
    ensure_dir(out_pdb.parent)

    content = docked_pdbqt.read_text(encoding="utf-8", errors="ignore")
    temp_pdbqt = out_pdb.with_suffix(".pose1.temp.pdbqt")

    try:
        if "MODEL " in content:
            m1_lines = []
            in_m1 = False
            for line in content.splitlines():
                if line.startswith("MODEL 1") or line.startswith("MODEL    1") or (not in_m1 and line.startswith("MODEL")):
                    in_m1 = True
                if in_m1:
                    m1_lines.append(line)
                    if line.startswith("ENDMDL"):
                        break
            if m1_lines:
                temp_pdbqt.write_text("\n".join(m1_lines), encoding="utf-8")
                dock_source = temp_pdbqt
            else:
                dock_source = docked_pdbqt
        else:
            dock_source = docked_pdbqt

        cmd = [obabel_bin, "-ipdbqt", str(dock_source), "-opdb", "-O", str(out_pdb)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out_pdb.exists() and out_pdb.stat().st_size > 0:
            return out_pdb
        return None
    finally:
        if temp_pdbqt.exists():
            temp_pdbqt.unlink(missing_ok=True)


def create_cocomplex_pdb(
    receptor_pdb: Path,
    peptide_pose_pdb: Path,
    out_complex_pdb: Path
) -> Path:
    """
    Merges target receptor PDB (Chain A) and docked peptide PDB (Chain B)
    into a single valid co-complex structure.
    """
    out_complex_pdb = Path(out_complex_pdb)
    ensure_dir(out_complex_pdb.parent)

    rec_st = gemmi.read_structure(str(receptor_pdb))
    pep_st = gemmi.read_structure(str(peptide_pose_pdb))

    # Detect existing receptor chain IDs to prevent collisions
    existing_chains = {chain.name for model in rec_st for chain in model}
    available_chains = [c for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in existing_chains]
    pep_chain_id = available_chains[0] if available_chains else "Z"

    # Assign non-colliding chain ID to peptide
    for model in pep_st:
        for chain in model:
            chain.name = pep_chain_id

    # Merge into receptor model 0
    if len(rec_st) > 0 and len(pep_st) > 0:
        for chain in pep_st[0]:
            rec_st[0].add_chain(chain)

    rec_st.write_minimal_pdb(str(out_complex_pdb))
    return out_complex_pdb


def generate_pymol_script(
    receptor_pdb: Path,
    top_complexes: List[Dict[str, Any]],
    script_path: Path
) -> Path:
    """
    Writes a one-click PyMOL visualization script (.pml) with styled representations:
    - Receptor shown in cartoon (grey/pale green)
    - Docked AMPs shown in cartoon + sticks
    - Binding pocket residues highlighted
    - Salt bridges & polar contacts shown as dashed lines
    """
    script_path = Path(script_path)
    ensure_dir(script_path.parent)

    lines = [
        "# PyMOL Visualization Script for ESM-AMP-Screen Hits",
        "reinitialize",
        "bg_color white",
        "set ray_shadows, 0",
        "set antialias, 2",
        f"load {receptor_pdb.resolve()}, receptor",
        "color gray80, receptor",
        "show cartoon, receptor",
        "set cartoon_transparency, 0.2, receptor",
        ""
    ]

    palette = ["cyan", "magenta", "orange", "yellow", "tv_blue", "ruby", "green"]

    for idx, hit in enumerate(top_complexes):
        c_path = hit["complex_pdb"]
        pep_id = hit["pep_id"]
        color = palette[idx % len(palette)]
        obj_name = f"hit_{idx+1}_{pep_id}"

        score_details = f"Vina: {hit['affinity']:.2f} kcal/mol"
        if "haddock_score" in hit:
            score_details += f", HADDOCK: {hit['haddock_score']:.1f}"
        if "ml_kd" in hit:
            score_details += f", ML-Kd: {hit['ml_kd']}"

        lines.extend([
            f"# Hit {idx+1}: {pep_id} ({score_details}, Composite: {hit.get('composite_score', 0):.2f})",
            f"load {c_path.resolve()}, {obj_name}",
            f"hide everything, {obj_name}",
            f"show cartoon, {obj_name} and chain B",
            f"show sticks, {obj_name} and chain B and not (name c,n,o)",
            f"color {color}, {obj_name} and chain B",
            f"select pocket_{idx+1}, (receptor within 4.5 of ({obj_name} and chain B))",
            f"show sticks, pocket_{idx+1} and not (name c,n,o)",
            f"color salmon, pocket_{idx+1}",
            f"distance contacts_{idx+1}, ({obj_name} and chain B and (name NZ,NH1,NH2)), (receptor and (name OD1,OD2,OE1,OE2)), 4.0",
            f"color yellow, contacts_{idx+1}",
            ""
        ])

    lines.extend([
        "zoom pocket_1, 6.0",
        "set dash_gap, 0.2",
        "set dash_width, 2.5",
        "message 'Loaded ESM-AMP-Screen top hits successfully!'"
    ])

    script_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated PyMOL visualization script: {script_path.name}")
    return script_path


def export_screening_report(
    summary_data: List[Dict[str, Any]],
    receptor_pdb: Path,
    out_dir: Path,
    top_n: int = 5,
    filename: str = "screening_leaderboard.csv",
    obabel_bin: str = "obabel"
) -> Tuple[Path, Path]:
    """
    Exports full CSV summary leaderboard, generates co-complexes for top N hits,
    and creates the PyMOL session script.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    complex_dir = ensure_dir(out_dir / "top_complexes")

    # Sort by Consensus Score if present in any hit, else Composite Score descending, then Vina affinity ascending
    if summary_data and any("consensus_score" in x for x in summary_data):
        summary_data.sort(key=lambda x: (x.get("consensus_score", -999), x.get("composite_score", -999)), reverse=True)
    else:
        summary_data.sort(key=lambda x: (x.get("composite_score", -999), -x.get("affinity", 0)), reverse=True)

    # Assign ranks
    for rank, row in enumerate(summary_data, 1):
        row["rank"] = rank
        if "estimated_kd" not in row or not row["estimated_kd"]:
            row["estimated_kd"] = estimate_kd(row["affinity"])

    # Generate co-complexes for top N hits
    top_hits = []
    for hit in summary_data[:top_n]:
        docked_pdbqt = hit.get("docked_pdbqt")
        if docked_pdbqt and Path(docked_pdbqt).exists():
            pep_id = hit["pep_id"]
            pose_pdb = complex_dir / f"{pep_id}_pose.pdb"
            convert_docked_pdbqt_to_pdb(docked_pdbqt, pose_pdb, obabel_bin=obabel_bin)
            
            orig_pdb = out_dir / "peptides_pdb" / f"{pep_id}.pdb"
            if orig_pdb.exists() and pose_pdb.exists():
                pristine_pose = complex_dir / f"{pep_id}_pristine_pose.pdb"
                pose_to_use = create_docked_pristine_pdb(orig_pdb, pose_pdb, pristine_pose)
            else:
                pose_to_use = pose_pdb

            if pose_to_use.exists():
                complex_pdb = complex_dir / f"complex_{pep_id}.pdb"
                create_cocomplex_pdb(receptor_pdb, pose_to_use, complex_pdb)
                hit["complex_pdb"] = complex_pdb
                top_hits.append(hit)

    df = pd.DataFrame(summary_data)
    csv_path = out_dir / filename
    df.to_csv(csv_path, index=False)
    logger.info(f"Screening leaderboard saved to {csv_path.name} ({len(df)} peptides ranked).")

    # Generate PyMOL script
    pml_path = out_dir / "view_hits.pml"
    if top_hits:
        generate_pymol_script(receptor_pdb, top_hits, pml_path)

    # Generate Standalone 3D HTML Report
    html_path = out_dir / "screening_report.html"
    try:
        generate_interactive_html_report(summary_data, receptor_pdb, html_path, top_n=top_n)
    except Exception as e:
        logger.warning(f"Could not generate HTML report: {e}")

    return csv_path, pml_path
