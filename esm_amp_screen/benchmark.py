"""
Benchmark and Validation Module for ESM-AMP-Screen.
Quantifies docking prediction accuracy against experimental co-crystal structures
or external tools (HADDOCK, AutoDock CrankPep, Rosetta FlexPepDock)
using official CAPRI (Critical Assessment of PRediction of Interactions) standards:
- Ligand Backbone RMSD (L-RMSD_bb)
- Ligand Heavy-Atom RMSD (L-RMSD_all)
- Interface RMSD (I-RMSD)
- Fraction of Native Contacts (F_nat)
- CAPRI Quality Classification (High, Medium, Acceptable, Incorrect)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np

from esm_amp_screen.utils import logger, ensure_dir


def kabsch_superposition(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes optimal rotation matrix R and translation vector t to superimpose
    movable coordinates p onto fixed coordinates q minimizing RMSD:
    q ~ p @ R.T + t
    """
    p_mean = p.mean(axis=0)
    q_mean = q.mean(axis=0)
    p_c = p - p_mean
    q_c = q - q_mean

    H = p_c.T @ q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    V = Vt.T.copy()
    V[:, -1] *= d

    R = V @ U.T
    t = q_mean - p_mean @ R.T
    return R, t


def extract_chain_atoms(
    structure: gemmi.Structure,
    chain_name: str,
    backbone_only: bool = False
) -> Dict[Tuple[int, str], np.ndarray]:
    """
    Extracts atom coordinates keyed by (residue_number, atom_name).
    """
    atoms = {}
    bb_names = {"N", "CA", "C", "O"}
    for model in structure:
        for chain in model:
            if chain.name == chain_name:
                for res in chain:
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        if backbone_only and atom.name not in bb_names:
                            continue
                        atoms[(res.seqid.num, atom.name)] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return atoms


def evaluate_docked_complex(
    docked_pdb: Path,
    reference_pdb: Path,
    docked_rec_chain: str = "A",
    docked_pep_chain: str = "B",
    ref_rec_chain: str = "A",
    ref_pep_chain: str = "B",
    contact_cutoff: float = 5.0,
    out_superimposed_pdb: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates a docked peptide-receptor complex against a reference co-crystal structure.
    Superimposes receptor backbones, transforms the peptide, and calculates CAPRI metrics.
    """
    docked_pdb = Path(docked_pdb)
    reference_pdb = Path(reference_pdb)

    st_docked = gemmi.read_structure(str(docked_pdb))
    st_ref = gemmi.read_structure(str(reference_pdb))

    # 1. Extract Receptor CA coordinates for structural superposition
    docked_rec_cas = {}
    ref_rec_cas = {}

    for model in st_docked:
        for chain in model:
            if chain.name == docked_rec_chain:
                for res in chain:
                    for atom in res:
                        if atom.name == "CA":
                            docked_rec_cas[res.seqid.num] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])

    for model in st_ref:
        for chain in model:
            if chain.name == ref_rec_chain:
                for res in chain:
                    for atom in res:
                        if atom.name == "CA":
                            ref_rec_cas[res.seqid.num] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])

    common_res = sorted(list(set(docked_rec_cas.keys()).intersection(ref_rec_cas.keys())))
    if len(common_res) < 10:
        raise ValueError(f"Insufficient common receptor CA atoms between docked ({len(docked_rec_cas)}) and reference ({len(ref_rec_cas)}).")

    p_rec = np.array([docked_rec_cas[r] for r in common_res])
    q_rec = np.array([ref_rec_cas[r] for r in common_res])

    # Superposition transformation matrix
    R, t = kabsch_superposition(p_rec, q_rec)
    rec_rmsd = float(np.sqrt(np.mean(np.sum(((p_rec @ R.T + t) - q_rec) ** 2, axis=1))))

    # 2. Extract Peptide Atoms and establish unambiguous correspondence
    docked_pep_atoms = extract_chain_atoms(st_docked, docked_pep_chain, backbone_only=False)
    ref_pep_atoms = extract_chain_atoms(st_ref, ref_pep_chain, backbone_only=False)

    # Establish reference-to-docked atom map
    common_pep_keys = set(docked_pep_atoms.keys()).intersection(ref_pep_atoms.keys())
    ref_to_docked_pep_map = {}

    if len(common_pep_keys) >= 4:
        for k in common_pep_keys:
            ref_to_docked_pep_map[k] = docked_pep_atoms[k]
    else:
        # Sequential residue alignment fallback (handles 1-based vs crystal numbering)
        docked_res_nums = sorted(list(set(k[0] for k in docked_pep_atoms.keys())))
        ref_res_nums = sorted(list(set(k[0] for k in ref_pep_atoms.keys())))
        min_res = min(len(docked_res_nums), len(ref_res_nums))
        for i in range(min_res):
            d_r = docked_res_nums[i]
            r_r = ref_res_nums[i]
            for (d_res, d_at), d_pos in docked_pep_atoms.items():
                if d_res == d_r and (r_r, d_at) in ref_pep_atoms:
                    ref_to_docked_pep_map[(r_r, d_at)] = d_pos

    matched_ref_keys = sorted(list(ref_to_docked_pep_map.keys()))
    if not matched_ref_keys:
        raise ValueError(
            f"Could not match peptide atoms between docked ({len(docked_pep_atoms)}) "
            f"and reference ({len(ref_pep_atoms)}) chains."
        )

    p_pep = np.array([ref_to_docked_pep_map[k] for k in matched_ref_keys])
    q_pep = np.array([ref_pep_atoms[k] for k in matched_ref_keys])
    bb_mask = np.array([k[1] in ("N", "CA", "C", "O") for k in matched_ref_keys])

    p_pep_transformed = p_pep @ R.T + t

    # Ligand RMSDs
    diff_all = p_pep_transformed - q_pep
    l_rmsd_all = float(np.sqrt(np.mean(np.sum(diff_all ** 2, axis=1))))

    if np.any(bb_mask):
        diff_bb = diff_all[bb_mask]
        l_rmsd_bb = float(np.sqrt(np.mean(np.sum(diff_bb ** 2, axis=1))))
    else:
        l_rmsd_bb = l_rmsd_all

    # 3. Fraction of Native Contacts (F_nat)
    # Find all native receptor-peptide atom contacts in reference structure within contact_cutoff
    ref_rec_atoms = extract_chain_atoms(st_ref, ref_rec_chain, backbone_only=False)
    native_pairs = set()

    for r_k, r_coord in ref_rec_atoms.items():
        for p_k, p_coord in ref_pep_atoms.items():
            if np.linalg.norm(r_coord - p_coord) <= contact_cutoff:
                native_pairs.add((r_k, p_k))

    f_nat = 0.0
    if native_pairs:
        recovered = 0
        for r_k, p_k in native_pairs:
            # Check if contact is maintained in transformed docked complex
            r_c = ref_rec_atoms.get(r_k)
            p_c = ref_to_docked_pep_map.get(p_k)
            if r_c is not None and p_c is not None:
                p_c_trans = p_c @ R.T + t
                if np.linalg.norm(r_c - p_c_trans) <= contact_cutoff:
                    recovered += 1
        f_nat = float(recovered / len(native_pairs))

    # 4. Interface RMSD (I-RMSD)
    # Definition (CAPRI standard): RMSD of backbone atoms (N, CA, C, O) of
    # interface residues from BOTH receptor AND peptide after superimposing receptors.
    # Interface = any residue with ≥1 heavy atom within 10 Å of any atom in the partner.
    # Reference: Mendez et al., Proteins 52:51-67 (2003).
    ref_rec_atoms_all = extract_chain_atoms(st_ref, ref_rec_chain, backbone_only=False)
    docked_rec_atoms_all = extract_chain_atoms(st_docked, docked_rec_chain, backbone_only=False)

    # Identify interface residue numbers in reference (receptor side: within 10Å of peptide)
    ref_rec_interface_res = set()
    ref_pep_q = np.array(list(ref_pep_atoms.values()))
    for (rnum, aname), r_coord in ref_rec_atoms_all.items():
        if ref_pep_q.size > 0:
            min_d = float(np.min(np.linalg.norm(ref_pep_q - r_coord, axis=1)))
            if min_d <= 10.0:
                ref_rec_interface_res.add(rnum)

    # Identify interface residue numbers in reference (peptide side: within 10Å of receptor)
    ref_pep_interface_res = set()
    ref_rec_q = np.array(list(ref_rec_atoms_all.values()))
    for (pnum, aname), p_coord in ref_pep_atoms.items():
        if ref_rec_q.size > 0:
            min_d = float(np.min(np.linalg.norm(ref_rec_q - p_coord, axis=1)))
            if min_d <= 10.0:
                ref_pep_interface_res.add(pnum)

    # Collect backbone atom pairs (ref vs docked, transformed) for interface residues
    bb_names = {"N", "CA", "C", "O"}
    irmsd_pts_ref = []
    irmsd_pts_docked = []

    # Receptor interface backbone atoms
    for (rnum, aname), ref_coord in ref_rec_atoms_all.items():
        if rnum in ref_rec_interface_res and aname in bb_names:
            docked_coord = docked_rec_atoms_all.get((rnum, aname))
            if docked_coord is not None:
                irmsd_pts_ref.append(ref_coord)
                irmsd_pts_docked.append(docked_coord @ R.T + t)   # apply receptor superposition

    # Peptide interface backbone atoms
    for (pnum, aname), ref_coord in ref_pep_atoms.items():
        if pnum in ref_pep_interface_res and aname in bb_names:
            docked_coord = ref_to_docked_pep_map.get((pnum, aname))
            if docked_coord is not None:
                irmsd_pts_ref.append(ref_coord)
                irmsd_pts_docked.append(docked_coord @ R.T + t)   # apply receptor superposition

    if len(irmsd_pts_ref) >= 3:
        diff_ir = np.array(irmsd_pts_docked) - np.array(irmsd_pts_ref)
        i_rmsd = float(np.sqrt(np.mean(np.sum(diff_ir ** 2, axis=1))))
    else:
        i_rmsd = l_rmsd_bb
        logger.debug("I-RMSD fallback: too few interface backbone atoms found.")

    # 5. CAPRI Quality Classification
    # Official CAPRI thresholds (Janin et al., Proteins 52:2-9 (2003)):
    # High (***):    F_nat ≥ 0.5  AND (L-RMSD ≤ 1.0 Å  OR I-RMSD ≤ 1.0 Å)
    # Medium (**):   F_nat ≥ 0.3  AND (L-RMSD ≤ 5.0 Å  OR I-RMSD ≤ 2.0 Å)
    # Acceptable (*):F_nat ≥ 0.1  AND (L-RMSD ≤ 10.0 Å OR I-RMSD ≤ 4.0 Å)
    # Incorrect:     otherwise
    # Note: each tier is exclusive from the previous (High takes precedence).
    if f_nat >= 0.5 and (l_rmsd_bb <= 1.0 or i_rmsd <= 1.0):
        capri_quality = "High (***)"
    elif f_nat >= 0.3 and (l_rmsd_bb <= 5.0 or i_rmsd <= 2.0):
        capri_quality = "Medium (**)"
    elif f_nat >= 0.1 and (l_rmsd_bb <= 10.0 or i_rmsd <= 4.0):
        capri_quality = "Acceptable (*)"
    else:
        capri_quality = "Incorrect"

    # Optional: Write Superimposed PDB
    if out_superimposed_pdb:
        out_superimposed_pdb = Path(out_superimposed_pdb)
        ensure_dir(out_superimposed_pdb.parent)
        for model in st_docked:
            for chain in model:
                for res in chain:
                    for atom in res:
                        pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                        new_pos = pos @ R.T + t
                        atom.pos = gemmi.Position(new_pos[0], new_pos[1], new_pos[2])
        st_docked.write_minimal_pdb(str(out_superimposed_pdb))
        logger.info(f"Superimposed complex structure written to {out_superimposed_pdb.name}")

    metrics = {
        "docked_complex": str(docked_pdb),
        "reference_crystal": str(reference_pdb),
        "receptor_ca_rmsd": round(rec_rmsd, 3),
        "l_rmsd_backbone": round(l_rmsd_bb, 3),
        "l_rmsd_all_atom": round(l_rmsd_all, 3),
        "i_rmsd": round(i_rmsd, 3),
        "f_nat": round(f_nat, 3),
        "f_nat_percent": round(f_nat * 100.0, 1),
        "capri_quality": capri_quality,
        "native_contacts_count": len(native_pairs)
    }
    return metrics
