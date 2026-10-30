"""
Target Protein Receptor Preparation Module.
Cleans receptor PDB, computes binding site grid box, and detects surface cavities.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from esm_amp_screen.utils import logger, ensure_dir, which_tool


def clean_receptor_pdb(
    input_pdb: Path,
    cleaned_pdb: Path,
    remove_hetero: bool = True,
    receptor_chain: Optional[str] = None
) -> Path:
    """
    Cleans target receptor PDB using gemmi:
    - Removes water molecules (HOH, WAT, DOD).
    - Optionally removes non-protein heteroatoms / crystallographic artifacts.
    - If receptor_chain is specified, isolates the requested receptor chain (stripping co-crystal ligands/peptides).
    - Preserves standard amino acid residues.
    """
    input_pdb = Path(input_pdb)
    cleaned_pdb = Path(cleaned_pdb)
    ensure_dir(cleaned_pdb.parent)

    doc = gemmi.read_structure(str(input_pdb))
    doc.setup_entities()
    try:
        if remove_hetero:
            doc.remove_ligands_and_waters()
        else:
            doc.remove_waters()
    except Exception:
        # Fallback manual cleaning of crystallographic waters and heteroatoms
        for model in doc:
            for chain in model:
                to_remove = []
                for i, res in enumerate(chain):
                    if res.name.upper() in ["HOH", "WAT", "DOD", "H2O", "TIP3"]:
                        to_remove.append(i)
                    elif remove_hetero and res.het_flag == 'H':
                        to_remove.append(i)
                for i in reversed(to_remove):
                    del chain[i]

    # If specific receptor chain requested, filter out extraneous chains (e.g. bound co-crystal peptides)
    if receptor_chain:
        for model in doc:
            chains_to_del = [chain.name for chain in model if chain.name != receptor_chain]
            for c_name in chains_to_del:
                del model[c_name]
    
    doc.write_minimal_pdb(str(cleaned_pdb))
    logger.info(f"Cleaned receptor saved to {cleaned_pdb.name}")
    return cleaned_pdb


HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}


def detect_surface_cavities(
    pdb_path: Path,
    max_pockets: int = 5,
    min_points: int = 8,
    padding: float = 8.0,
    min_pocket_distance: float = 8.0
) -> List[Dict[str, Any]]:
    """
    Detects and profiles surface cavities and druggable binding pockets on a target receptor:
    1. Extracts receptor heavy atoms and generates a 3D probe grid (1.8 Å spacing).
    2. Identifies probe points in the solvent-accessible surface layer (2.5 Å <= d <= 6.2 Å).
    3. Evaluates 7-ray Protein-Solvent-Protein (PSP / LIGSITE standard) directional enclosure.
    4. Clusters enclosed cavity points using DBSCAN (eps=2.5 Å, min_samples=6) to identify discrete pockets.
    5. Calculates cavity volume (Å³), mean enclosure depth, hydrophobic ratio, and 0-100 Druggability Score.
    6. Filters for spatially distinct pockets (center distance >= min_pocket_distance) and ranks them.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Receptor PDB not found for cavity detection: {pdb_path}")

    st = gemmi.read_structure(str(pdb_path))
    coords = []
    res_id_map = []
    res_name_map = []

    for model in st:
        for chain in model:
            for res in chain:
                if res.name.upper() in ("HOH", "WAT", "TIP3", "SOL", "DOD"):
                    continue
                for atom in res:
                    if atom.element.name != "H":
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        res_id_map.append(f"{chain.name}:{res.seqid.num}")
                        res_name_map.append(f"{chain.name}:{res.seqid.num}:{res.name}")

    if not coords:
        return []

    prot_coords = np.array(coords)
    tree = cKDTree(prot_coords)

    spacing = 1.8
    min_c = prot_coords.min(axis=0) - 1.5
    max_c = prot_coords.max(axis=0) + 1.5
    gx = np.arange(min_c[0], max_c[0], spacing)
    gy = np.arange(min_c[1], max_c[1], spacing)
    gz = np.arange(min_c[2], max_c[2], spacing)
    grid_pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)

    # 1. Distance filter: keep points in solvent-accessible surface layer
    dists, _ = tree.query(grid_pts, k=1)
    surf_mask = (dists >= 2.5) & (dists <= 6.2)
    surf_pts = grid_pts[surf_mask]
    if len(surf_pts) == 0:
        return []

    # 2. 7-ray PSP (Protein-Solvent-Protein) directional concavity evaluation
    sqrt3 = np.sqrt(3.0)
    rays = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3],
        [1.0 / sqrt3, 1.0 / sqrt3, -1.0 / sqrt3],
        [1.0 / sqrt3, -1.0 / sqrt3, 1.0 / sqrt3],
        [-1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3]
    ])

    nbrs_list = tree.query_ball_point(surf_pts, r=9.5)
    cavity_mask = []
    enclosure_scores = []

    for pt, nbrs in zip(surf_pts, nbrs_list):
        if len(nbrs) < 6:
            cavity_mask.append(False)
            enclosure_scores.append(0)
            continue
        disp = prot_coords[nbrs] - pt
        d = np.linalg.norm(disp, axis=1)
        valid = (d >= 2.2) & (d <= 9.5)
        if not np.any(valid):
            cavity_mask.append(False)
            enclosure_scores.append(0)
            continue
        u = disp[valid] / d[valid, None]
        projs = np.dot(u, rays.T)
        pos = np.any(projs > 0.60, axis=0)
        neg = np.any(projs < -0.60, axis=0)
        psp = int(np.sum(pos & neg))
        enclosure_scores.append(psp)
        cavity_mask.append(psp >= 2)

    cavity_mask = np.array(cavity_mask)
    cav_pts = surf_pts[cavity_mask]
    cav_enc = np.array(enclosure_scores)[cavity_mask]

    # Graceful fallback if surface concavity is unusually shallow
    if len(cav_pts) < min_points:
        cavity_mask_fb = np.array([e >= 1 for e in enclosure_scores])
        cav_pts = surf_pts[cavity_mask_fb]
        cav_enc = np.array(enclosure_scores)[cavity_mask_fb]
        if len(cav_pts) < min_points:
            return []

    # 3. Cluster cavity points with DBSCAN
    db = DBSCAN(eps=2.5, min_samples=6).fit(cav_pts)
    labels = db.labels_
    unique_labels = [l for l in set(labels) if l != -1]
    if not unique_labels:
        return []

    clusters = []
    for l in unique_labels:
        idx = np.where(labels == l)[0]
        c_pts = cav_pts[idx]
        c_enc = cav_enc[idx]
        if len(c_pts) < min_points:
            continue

        vol = len(c_pts) * (spacing ** 3)
        mean_enc = float(np.mean(c_enc))
        center = c_pts.mean(axis=0)

        # Surrounding lining residues within 4.5 Å
        near_indices = set()
        for idxs in tree.query_ball_point(c_pts, r=4.5):
            near_indices.update(idxs)
        c_res = sorted(list(set(res_id_map[i] for i in near_indices)))
        c_names = sorted(list(set(res_name_map[i] for i in near_indices)))

        # Hydrophobic character
        hydro_count = sum(1 for name in c_names if name.split(":")[-1].upper() in HYDROPHOBIC_RESIDUES)
        hydro_ratio = hydro_count / max(1, len(c_names))

        # Druggability score (0 to 100): Volume (40) + Enclosure Depth (35) + Hydrophobicity (25)
        vol_score = min(40.0, 40.0 * (vol / 800.0))
        enc_score = min(35.0, 35.0 * max(0.0, (mean_enc - 1.5) / 3.0))
        hydro_score = min(25.0, 25.0 * (hydro_ratio / 0.50))
        druggability = round(min(100.0, vol_score + enc_score + hydro_score), 1)

        if druggability >= 75.0:
            d_class = "High Druggability (Primary Cleft)"
        elif druggability >= 50.0:
            d_class = "Moderate Druggability (Secondary Pocket)"
        else:
            d_class = "Low Druggability (Shallow Groove)"

        span = c_pts.max(axis=0) - c_pts.min(axis=0) + (2 * padding)
        dims = np.clip(span, 26.0, 50.0)

        clusters.append({
            "druggability_score": druggability,
            "druggability_class": d_class,
            "volume_angstrom3": round(float(vol), 1),
            "num_points": int(len(c_pts)),
            "center_x": round(float(center[0]), 3),
            "center_y": round(float(center[1]), 3),
            "center_z": round(float(center[2]), 3),
            "size_x": round(float(dims[0]), 1),
            "size_y": round(float(dims[1]), 1),
            "size_z": round(float(dims[2]), 1),
            "enclosure_score": round(mean_enc, 2),
            "hydrophobic_ratio": round(hydro_ratio, 3),
            "contact_residues": c_res,
            "contact_residues_formatted": c_names,
            "center_vec": center
        })

    clusters.sort(key=lambda x: x["druggability_score"], reverse=True)

    # Spatial distinctiveness filter (non-maximum suppression by distance)
    distinct_pockets = []
    for c in clusters:
        c_pos = c["center_vec"]
        too_close = False
        for s in distinct_pockets:
            if np.linalg.norm(c_pos - s["center_vec"]) < min_pocket_distance:
                too_close = True
                break
        if not too_close:
            distinct_pockets.append(c)
        if len(distinct_pockets) >= max_pockets:
            break

    for i, p in enumerate(distinct_pockets, 1):
        p["pocket_id"] = i
        p.pop("center_vec", None)

    return distinct_pockets


def calculate_grid_box(
    pdb_path: Path,
    center: Optional[Tuple[float, float, float]] = None,
    size: Optional[Tuple[float, float, float]] = None,
    pocket_residues: Optional[List[str]] = None,
    auto_pocket: bool = True,
    padding: float = 8.0,
    max_pockets: int = 5
) -> Dict[str, Any]:
    """
    Calculates center (cx, cy, cz) and dimensions (sx, sy, sz) for peptide search box:
    - If center and size are given, uses them directly.
    - If pocket residues are specified (e.g. ['A:45', 'A:48']), centers on their coordinates.
    - If auto_pocket is True and neither center nor pocket_residues are given,
      automatically detects the top surface cavity using LIGSITE-style 7-ray PSP concavity.
    - Otherwise, centers on receptor center-of-mass with automatic bounding dimensions.
    """
    if center is not None and size is not None:
        return {
            "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
            "size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2])
        }

    st = gemmi.read_structure(str(pdb_path))
    coords = []

    if pocket_residues:
        # Format: ["A:45", "B:102"] or ["45", "102"]
        target_res_keys = set()
        for r in pocket_residues:
            parts = r.strip().split(":")
            if len(parts) >= 2:
                try:
                    target_res_keys.add((parts[0].strip(), int(parts[1])))
                except ValueError:
                    continue
            elif len(parts) == 1:
                try:
                    target_res_keys.add(("", int(parts[0])))
                except ValueError:
                    continue

        for model in st:
            for chain in model:
                for res in chain:
                    if (chain.name, res.seqid.num) in target_res_keys or ("", res.seqid.num) in target_res_keys:
                        for atom in res:
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        
        if not coords:
            logger.warning("None of the specified pocket residues were found in PDB! Reverting to auto-pocket / whole-protein.")

    # Automated geometric cavity detection if neither center nor pocket residues provided
    if not coords and auto_pocket:
        try:
            detected_pockets = detect_surface_cavities(pdb_path, max_pockets=max_pockets, padding=padding)
            if detected_pockets:
                p1 = detected_pockets[0]
                logger.info(
                    f"Auto-Pocket Finder: Discovered {len(detected_pockets)} distinct surface cavity/pocket(s). "
                    f"Targeting Pocket #1 (Druggability: {p1['druggability_score']}/100, {p1['volume_angstrom3']} Å³) centered at "
                    f"({p1['center_x']}, {p1['center_y']}, {p1['center_z']}) with dimensions "
                    f"({p1['size_x']} x {p1['size_y']} x {p1['size_z']}) Å "
                    f"enclosed by {len(p1['contact_residues'])} residues."
                )
                return {
                    "center_x": p1["center_x"],
                    "center_y": p1["center_y"],
                    "center_z": p1["center_z"],
                    "size_x": p1["size_x"],
                    "size_y": p1["size_y"],
                    "size_z": p1["size_z"],
                    "detected_pocket_residues": p1["contact_residues"],
                    "detected_pockets": detected_pockets
                }
            else:
                logger.info("Auto-Pocket Finder found no significant cavities; defaulting to whole-protein center.")
        except Exception as e:
            logger.warning(f"Auto-Pocket detection failed: {e}; defaulting to whole-protein center.")

    if not coords:
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    arr = np.array(coords)
    min_pt = arr.min(axis=0)
    max_pt = arr.max(axis=0)
    center_pt = (min_pt + max_pt) / 2.0
    
    if size is not None:
        dims = np.array([float(s) for s in size])
    else:
        dims = (max_pt - min_pt) + (2 * padding)
        # Ensure box is sufficiently large for peptide span (28 to 60 Å)
        dims = np.clip(dims, 28.0, 60.0)

    grid = {
        "center_x": round(float(center_pt[0]), 3),
        "center_y": round(float(center_pt[1]), 3),
        "center_z": round(float(center_pt[2]), 3),
        "size_x": round(float(dims[0]), 1),
        "size_y": round(float(dims[1]), 1),
        "size_z": round(float(dims[2]), 1),
    }
    logger.info(
        f"Search Box: Center=({grid['center_x']}, {grid['center_y']}, {grid['center_z']}), "
        f"Size=({grid['size_x']} x {grid['size_y']} x {grid['size_z']}) Å"
    )
    return grid


def prepare_receptor_pdbqt(
    input_pdb: Path,
    out_dir: Path,
    obabel_bin: str = "obabel"
) -> Path:
    """
    Converts receptor PDB to PDBQT format using OpenBabel (-xr -p 7.4).
    Adds polar hydrogens (HD), assigns partial charges, and keeps receptor rigid.
    """
    input_pdb = Path(input_pdb)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    out_pdbqt = out_dir / f"{input_pdb.stem}.pdbqt"

    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        content = out_pdbqt.read_text(encoding="utf-8", errors="ignore")
        if " HD" in content:
            logger.info(f"Receptor PDBQT already exists with polar hydrogens: {out_pdbqt.name}")
            return out_pdbqt

    logger.info(f"Preparing receptor PDBQT with polar hydrogens at pH 7.4: {input_pdb.name} -> {out_pdbqt.name}")
    cmd = [obabel_bin, str(input_pdb), "-O", str(out_pdbqt), "-xr", "-p", "7.4"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not out_pdbqt.exists() or out_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            f"Failed to generate receptor PDBQT with OpenBabel.\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )

    logger.info(f"Receptor PDBQT generated successfully ({out_pdbqt.stat().st_size} bytes)")
    return out_pdbqt
