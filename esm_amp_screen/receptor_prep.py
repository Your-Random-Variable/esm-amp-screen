"""
Target Protein Receptor Preparation Module.
Cleans receptor PDB, computes binding site grid box, and converts to PDBQT.
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


def detect_surface_cavities(
    pdb_path: Path,
    max_pockets: int = 3,
    min_points: int = 15,
    padding: float = 8.0
) -> List[Dict[str, Any]]:
    """
    Detects surface cavities and druggable binding pockets on a target receptor:
    1. Extracts receptor heavy atoms and generates a 3D probe grid (2.0 Å spacing).
    2. Identifies probe points in the solvent-accessible surface layer (2.6 Å <= d <= 6.5 Å).
    3. Evaluates directional enclosure / concavity across the 6 Cartesian axes (+/- X, +/- Y, +/- Z).
    4. Clusters enclosed cavity points using DBSCAN to identify discrete surface pockets.
    5. Returns ranked pockets with center coordinates, bounding dimensions, and surrounding residues.
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
                for atom in res:
                    if atom.element.name != "H":
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        res_id_map.append(f"{chain.name}:{res.seqid.num}")
                        res_name_map.append(f"{chain.name}:{res.seqid.num}:{res.name}")

    if not coords:
        return []

    prot_coords = np.array(coords)
    tree = cKDTree(prot_coords)

    min_c = prot_coords.min(axis=0) - 2.0
    max_c = prot_coords.max(axis=0) + 2.0
    spacing = 2.0
    gx = np.arange(min_c[0], max_c[0], spacing)
    gy = np.arange(min_c[1], max_c[1], spacing)
    gz = np.arange(min_c[2], max_c[2], spacing)
    grid_pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)

    # 1. Distance filter: keep points in solvent-accessible surface layer
    dists, _ = tree.query(grid_pts, k=1)
    surf_mask = (dists >= 2.6) & (dists <= 6.5)
    surf_pts = grid_pts[surf_mask]
    if len(surf_pts) == 0:
        return []

    # 2. Concavity check: evaluate directional enclosure within 8.0 Å
    nbrs_list = tree.query_ball_point(surf_pts, r=8.0)
    cavity_mask = []
    for pt, nbrs in zip(surf_pts, nbrs_list):
        if len(nbrs) < 8:
            cavity_mask.append(False)
            continue
        diffs = prot_coords[nbrs] - pt
        enc = sum([
            np.any(diffs[:, 0] > 1.5), np.any(diffs[:, 0] < -1.5),
            np.any(diffs[:, 1] > 1.5), np.any(diffs[:, 1] < -1.5),
            np.any(diffs[:, 2] > 1.5), np.any(diffs[:, 2] < -1.5)
        ])
        cavity_mask.append(enc >= 4)

    cavity_pts = surf_pts[np.array(cavity_mask)]
    if len(cavity_pts) < min_points:
        return []

    # 3. Cluster cavity points with DBSCAN
    db = DBSCAN(eps=4.0, min_samples=6).fit(cavity_pts)
    labels = db.labels_
    unique_labels = [l for l in set(labels) if l != -1]
    if not unique_labels:
        return []

    clusters = []
    for l in unique_labels:
        c_pts = cavity_pts[labels == l]
        if len(c_pts) >= min_points:
            clusters.append((len(c_pts), c_pts.mean(axis=0), c_pts))
    clusters.sort(key=lambda x: x[0], reverse=True)

    results = []
    for i, (size, center, c_pts) in enumerate(clusters[:max_pockets]):
        span = c_pts.max(axis=0) - c_pts.min(axis=0) + (2 * padding)
        dims = np.clip(span, 26.0, 50.0)

        # Contact residues within 4.5 Å of cavity probes
        near_indices = set()
        for idxs in tree.query_ball_point(c_pts, r=4.5):
            near_indices.update(idxs)
        c_res = sorted(list(set(res_id_map[idx] for idx in near_indices)))
        c_names = sorted(list(set(res_name_map[idx] for idx in near_indices)))

        results.append({
            "pocket_id": i + 1,
            "num_points": int(size),
            "center_x": round(float(center[0]), 3),
            "center_y": round(float(center[1]), 3),
            "center_z": round(float(center[2]), 3),
            "size_x": round(float(dims[0]), 1),
            "size_y": round(float(dims[1]), 1),
            "size_z": round(float(dims[2]), 1),
            "contact_residues": c_res,
            "contact_residues_formatted": c_names
        })

    return results


def calculate_grid_box(
    pdb_path: Path,
    center: Optional[Tuple[float, float, float]] = None,
    size: Optional[Tuple[float, float, float]] = None,
    pocket_residues: Optional[List[str]] = None,
    auto_pocket: bool = True,
    padding: float = 8.0
) -> Dict[str, Any]:
    """
    Calculates center (cx, cy, cz) and dimensions (sx, sy, sz) for Vina search box:
    - If center and size are given, uses them directly.
    - If pocket residues are specified (e.g. ['A:45', 'A:48']), centers on their coordinates.
    - If auto_pocket is True and neither center nor pocket_residues are given,
      automatically detects the top surface cavity using LIGSITE-style grid concavity.
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
            detected_pockets = detect_surface_cavities(pdb_path, max_pockets=3, padding=padding)
            if detected_pockets:
                p1 = detected_pockets[0]
                logger.info(
                    f"Auto-Pocket Finder: Discovered {len(detected_pockets)} surface cavity/pocket(s). "
                    f"Targeting Pocket #1 ({p1['num_points']} probe points) centered at "
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
