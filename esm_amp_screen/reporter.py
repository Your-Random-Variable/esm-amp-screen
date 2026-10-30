"""
Reporting & Visualization Module.
Exports ranked CSV leaderboard, target-peptide co-complex PDB structures,
and interactive PyMOL visualization scripts.
"""

import re
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


def count_docked_modes(docked_pdbqt: Path) -> int:
    """Counts the number of distinct docked MODEL blocks in a PDBQT file."""
    if not docked_pdbqt or not Path(docked_pdbqt).exists():
        return 0
    content = Path(docked_pdbqt).read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"^MODEL\b", content, re.MULTILINE))


def convert_docked_pdbqt_to_pdb(
    docked_pdbqt: Path,
    out_pdb: Path,
    obabel_bin: str = "obabel",
    model_idx: int = 1
) -> Optional[Path]:
    """Converts a specific pose (Model N, default: Model 1) of a docked PDBQT file to PDB using OpenBabel."""
    out_pdb = Path(out_pdb)
    docked_pdbqt = Path(docked_pdbqt)
    ensure_dir(out_pdb.parent)

    content = docked_pdbqt.read_text(encoding="utf-8", errors="ignore")
    temp_pdbqt = out_pdb.with_suffix(f".pose{model_idx}.temp.pdbqt")

    try:
        if "MODEL " in content or "MODEL\t" in content:
            in_target_model = False
            current_model_idx = 0
            m_lines = []
            for line in content.splitlines():
                if line.startswith("MODEL"):
                    current_model_idx += 1
                    if current_model_idx == model_idx:
                        in_target_model = True
                        m_lines.append(line)
                        continue
                if in_target_model:
                    m_lines.append(line)
                    if line.startswith("ENDMDL"):
                        break
            if m_lines:
                temp_pdbqt.write_text("\n".join(m_lines), encoding="utf-8")
                dock_source = temp_pdbqt
            else:
                return None
        else:
            if model_idx > 1:
                return None
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
        c_path = Path(hit["complex_pdb"])
        pep_id = hit["pep_id"]
        color = palette[idx % len(palette)]
        obj_name = f"hit_{idx+1}_{pep_id}"

        score_details = f"Docking: {hit['affinity']:.2f} kcal/mol"
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

    # Sort by Consensus Score if present in any hit (ensuring rescored hits precede unrescored hits)
    if summary_data and any("consensus_score" in x for x in summary_data):
        def sort_key(x):
            has_cs = "consensus_score" in x and x["consensus_score"] is not None
            return (1 if has_cs else 0, x.get("consensus_score", -999999), x.get("composite_score", -999))
        summary_data.sort(key=sort_key, reverse=True)
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
        pep_id = hit["pep_id"]
        target_cpx = complex_dir / f"complex_{pep_id}.pdb"
        complex_pdb = hit.get("complex_pdb")
        docked_pdb = hit.get("docked_pdb")
        docked_pdbqt = hit.get("docked_pdbqt")

        if complex_pdb and Path(complex_pdb).exists():
            if Path(complex_pdb).resolve() != target_cpx.resolve():
                import shutil
                shutil.copy2(complex_pdb, target_cpx)
            hit["complex_pdb"] = target_cpx
            top_hits.append(hit)
        elif docked_pdb and Path(docked_pdb).exists():
            create_cocomplex_pdb(receptor_pdb, Path(docked_pdb), target_cpx)
            hit["complex_pdb"] = target_cpx
            top_hits.append(hit)
        elif docked_pdbqt and Path(docked_pdbqt).exists():
            pose_pdb = complex_dir / f"{pep_id}_pose.pdb"
            convert_docked_pdbqt_to_pdb(docked_pdbqt, pose_pdb, obabel_bin=obabel_bin)
            
            orig_pdb = out_dir / "peptides_pdb" / f"{pep_id}.pdb"
            if orig_pdb.exists() and pose_pdb.exists():
                pristine_pose = complex_dir / f"{pep_id}_pristine_pose.pdb"
                pose_to_use = create_docked_pristine_pdb(orig_pdb, pose_pdb, pristine_pose)
            else:
                pose_to_use = pose_pdb

            if pose_to_use.exists():
                create_cocomplex_pdb(receptor_pdb, pose_to_use, target_cpx)
                hit["complex_pdb"] = target_cpx
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


def generate_pockets_pymol_script(
    receptor_pdb: Path,
    pockets: List[Dict[str, Any]],
    script_path: Path,
    docked_complexes: Optional[Dict[int, Path]] = None
) -> Path:
    """
    Generates a color-coded PyMOL visualization script (.pml) for multi-pocket blind docking:
    - Receptor shown in transparent cartoon (gray).
    - Distinct vibrant colors for top pockets (Pocket 1: Firebrick/Red, Pocket 2: Green, Pocket 3: Marine/Cyan, Pocket 4: Orange/Yellow, Pocket 5: Magenta).
    - Pocket lining residues shown in sticks.
    - Pocket geometric centers marked with pseudoatoms & labeled spheres.
    - Docked peptide poses / co-complexes loaded and highlighted if provided.
    """
    script_path = Path(script_path)
    ensure_dir(script_path.parent)

    colors = [
        "firebrick",    # Pocket 1 (Primary)
        "forest",       # Pocket 2 (Secondary)
        "marine",       # Pocket 3
        "brightorange", # Pocket 4
        "warmpink",     # Pocket 5
        "deepteal",
        "violet",
        "wheat"
    ]

    lines = [
        "# PyMOL Visualization Script for ESM-AMP-Screen Multi-Pocket Cavities",
        "reinitialize",
        "bg_color white",
        "set ray_shadows, 0",
        "set antialias, 2",
        f"load {receptor_pdb.resolve()}, receptor",
        "hide everything, receptor",
        "show cartoon, receptor",
        "color gray80, receptor",
        "set cartoon_transparency, 0.25, receptor",
        ""
    ]

    for i, p in enumerate(pockets):
        p_id = p.get("pocket_id", i + 1)
        color = colors[(p_id - 1) % len(colors)]
        cx, cy, cz = p["center_x"], p["center_y"], p["center_z"]
        score = p.get("druggability_score", 0.0)
        vol = p.get("volume_angstrom3", 0.0)

        # Contact residue selection
        c_res = p.get("contact_residues", [])
        if c_res:
            res_nums = []
            for r in c_res:
                parts = r.split(":")
                num = parts[-1]
                if num.lstrip("-").isdigit():
                    res_nums.append(num)
            if res_nums:
                res_str = "+".join(res_nums)
                lines.extend([
                    f"# Pocket {p_id} (Druggability: {score}/100, Vol: {vol} A^3)",
                    f"select pocket_{p_id}_res, receptor and resi {res_str}",
                    f"show sticks, pocket_{p_id}_res",
                    f"color {color}, pocket_{p_id}_res",
                    f"set stick_radius, 0.22, pocket_{p_id}_res",
                ])

        # Center pseudoatom
        lines.extend([
            f"pseudoatom p{p_id}_center, pos=[{cx:.2f}, {cy:.2f}, {cz:.2f}], label='P{p_id} ({score})'",
            f"show spheres, p{p_id}_center",
            f"set sphere_scale, 1.2, p{p_id}_center",
            f"color {color}, p{p_id}_center",
            f"set label_size, 14, p{p_id}_center",
            f"set label_color, {color}, p{p_id}_center",
            ""
        ])

        # Load docked complex or peptide pose if provided
        if docked_complexes and p_id in docked_complexes:
            cpx_file = Path(docked_complexes[p_id])
            if cpx_file.exists():
                lines.extend([
                    f"load {cpx_file.resolve()}, pocket_{p_id}_docked",
                    f"show cartoon, pocket_{p_id}_docked and chain B",
                    f"show sticks, pocket_{p_id}_docked and chain B",
                    f"color {color}, pocket_{p_id}_docked and chain B",
                    f"set stick_radius, 0.28, pocket_{p_id}_docked and chain B",
                    ""
                ])

    lines.extend([
        "zoom receptor",
        "orient",
        "set seq_view, 1"
    ])

    script_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated multi-pocket PyMOL visualization script: {script_path.name}")
    return script_path
