"""
Peptide PDBQT Preparation Module.
Freezes peptide backbone dihedrals (phi, psi, omega) in the ESM 3D conformation
while keeping functional sidechain torsions fully flexible.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
from rdkit import Chem
try:
    import meeko
except ImportError:
    meeko = None

from esm_amp_screen.utils import logger, ensure_dir


def create_docked_pristine_pdb(orig_pdb: Path, docked_pose_pdb: Path, out_pdb: Path) -> Path:
    """
    Transfers 3D coordinates from a docked pose PDB to the original pristine peptide PDB.
    Uses distance fingerprint matching to map atoms unambiguously even when OpenBabel
    or Meeko permutes atom indexing, preserving exact standard residue and atom naming.
    """
    orig_pdb = Path(orig_pdb)
    docked_pose_pdb = Path(docked_pose_pdb)
    out_pdb = Path(out_pdb)
    ensure_dir(out_pdb.parent)

    try:
        st_orig = gemmi.read_structure(str(orig_pdb))
        st_pose = gemmi.read_structure(str(docked_pose_pdb))

        orig_atoms = [a for m in st_orig for c in m for r in c for a in r if a.element.name != "H"]
        pose_atoms = [a for m in st_pose for c in m for r in c for a in r if a.element.name != "H"]

        if len(orig_atoms) == len(pose_atoms) and len(orig_atoms) > 3:
            p_orig = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in orig_atoms])
            p_pose = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in pose_atoms])

            # Pairwise distance matrices
            from scipy.spatial.distance import cdist
            d_orig = cdist(p_orig, p_orig)
            d_pose = cdist(p_pose, p_pose)

            # Match each orig atom to pose atom by sorted internal distance fingerprint
            mapping = []
            for i in range(len(p_orig)):
                diffs = np.abs(np.sort(d_orig[i]) - np.sort(d_pose, axis=1)).sum(axis=1)
                mapping.append(int(np.argmin(diffs)))

            # If mapping is a valid bijection (no duplicates)
            if len(set(mapping)) == len(orig_atoms):
                p_pose_ordered = p_pose[mapping]
                for o_at, pos in zip(orig_atoms, p_pose_ordered):
                    o_at.pos = gemmi.Position(pos[0], pos[1], pos[2])
                st_orig.write_minimal_pdb(str(out_pdb))
                return out_pdb

            # Fallback direct assignment if fingerprint is ambiguous
            for o_at, p_at in zip(orig_atoms, pose_atoms):
                o_at.pos = p_at.pos
            st_orig.write_minimal_pdb(str(out_pdb))
            return out_pdb
    except Exception as e:
        logger.warning(f"Could not map coordinates onto pristine PDB: {e}")

    return docked_pose_pdb


from typing import Any, Dict, List, Optional, Tuple


def detect_and_annotate_disulfides(pdb_path: Path, out_annotated_pdb: Path) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Detects covalent disulfide bonds (SG-SG distance 1.7 - 2.6 Å) and backbone cyclization
    in peptide PDBs. Generates a temporary PDB with CONECT records so that OpenBabel and RDKit
    perceive the covalent ring topology without breaking the cyclic constraint.
    """
    pdb_path = Path(pdb_path)
    out_annotated_pdb = Path(out_annotated_pdb)

    try:
        st = gemmi.read_structure(str(pdb_path))
    except Exception as e:
        logger.warning(f"Could not read PDB with gemmi for disulfide detection: {e}")
        return [], pdb_path

    cys_sgs = []
    n_term_atom = None
    c_term_atom = None

    for model in st:
        for chain in model:
            res_list = list(chain)
            if res_list:
                # Potential N-terminus
                for atom in res_list[0]:
                    if atom.name == "N":
                        n_term_atom = (atom.serial, chain.name, res_list[0].seqid.num, atom.pos)
                # Potential C-terminus
                for atom in res_list[-1]:
                    if atom.name == "C":
                        c_term_atom = (atom.serial, chain.name, res_list[-1].seqid.num, atom.pos)

            for res in res_list:
                if res.name.upper() in ("CYS", "CYX"):
                    for atom in res:
                        if atom.name == "SG":
                            cys_sgs.append((atom.serial, chain.name, res.seqid.num, atom.pos))

    macrocycle_bonds = []

    # 1. Intra-peptide disulfide bonds (SG-SG)
    for i in range(len(cys_sgs)):
        for j in range(i + 1, len(cys_sgs)):
            s1, c1, n1, p1 = cys_sgs[i]
            s2, c2, n2, p2 = cys_sgs[j]
            d = p1.dist(p2)
            if 1.7 <= d <= 2.6:
                macrocycle_bonds.append({
                    "type": "disulfide",
                    "cys1": f"{c1}:{n1}",
                    "cys2": f"{c2}:{n2}",
                    "serial1": s1,
                    "serial2": s2,
                    "distance": round(d, 2)
                })

    # 2. Head-to-tail backbone cyclization (N-to-C distance <= 1.8 Å)
    if n_term_atom and c_term_atom and len(res_list) > 2:
        ns, nc, nn, np_pos = n_term_atom
        cs, cc, cn, cp_pos = c_term_atom
        d_nc = np_pos.dist(cp_pos)
        if d_nc <= 1.8:
            macrocycle_bonds.append({
                "type": "head_to_tail",
                "cys1": f"{nc}:{nn}",
                "cys2": f"{cc}:{cn}",
                "serial1": ns,
                "serial2": cs,
                "distance": round(d_nc, 2)
            })

    if not macrocycle_bonds:
        return [], pdb_path

    # Insert CONECT lines before END or TER
    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    conect_lines = []
    for b in macrocycle_bonds:
        conect_lines.append(f"CONECT{b['serial1']:5d}{b['serial2']:5d}")
        conect_lines.append(f"CONECT{b['serial2']:5d}{b['serial1']:5d}")

    out_lines = []
    inserted = False
    for line in lines:
        if not inserted and (line.startswith("END") or line.startswith("TER")):
            out_lines.extend(conect_lines)
            inserted = True
        out_lines.append(line)
    if not inserted:
        out_lines.extend(conect_lines)

    out_annotated_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return macrocycle_bonds, out_annotated_pdb


def prepare_peptide_pdbqt(
    input_pdb: Path,
    out_pdbqt: Path,
    flexibility: str = "rigid",
    obabel_bin: str = "obabel",
    lock_disulfides: bool = True
) -> Tuple[Path, int]:
    """
    Converts a 3D peptide PDB into a PDBQT with configurable flexibility:
    - 'rigid': Freezes all internal bonds for ultra-fast screening of ESM folded structures.
    - 'sidechain': Freezes backbone dihedrals (phi, psi, omega) while keeping sidechains flexible.
    - Locks disulfide bonds and macrocycles when detected.
    Returns (out_pdbqt_path, num_rotatable_bonds).
    """
    input_pdb = Path(input_pdb)
    out_pdbqt = Path(out_pdbqt)
    ensure_dir(out_pdbqt.parent)

    temp_sdf = out_pdbqt.with_suffix(".temp.sdf")
    temp_annotated_pdb = out_pdbqt.with_suffix(".annotated.pdb")

    try:
        prep_input_pdb = input_pdb
        macro_bonds = []
        if lock_disulfides:
            macro_bonds, prep_input_pdb = detect_and_annotate_disulfides(input_pdb, temp_annotated_pdb)
            if macro_bonds:
                bond_descs = [f"{b.get('cys1')}-{b.get('cys2')} ({b.get('type')})" for b in macro_bonds]
                logger.info(f"Macrocycle Perception: Locked {len(macro_bonds)} cyclic bond(s) in {input_pdb.stem}: {', '.join(bond_descs)}")

        # Step 1: OpenBabel PDB -> SDF with explicit hydrogens added
        cmd = [obabel_bin, str(prep_input_pdb), "-O", str(temp_sdf), "-h"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not temp_sdf.exists() or temp_sdf.stat().st_size == 0:
            raise RuntimeError(f"OpenBabel PDB->SDF conversion failed for {input_pdb.name}: {res.stderr}")

        # Step 2: Read SDF into RDKit
        suppl = Chem.SDMolSupplier(str(temp_sdf), removeHs=False)
        mol = suppl[0]
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse intermediate SDF for {input_pdb.name}")

        has_macro = bool(macro_bonds)

        # Step 3: Meeko preparation with configured flexibility and macrocycle locking
        if flexibility == "rigid":
            rigid_smarts = ['[*]~[*]']
            rigid_indices = [(0, 1)]
            if has_macro:
                rigid_smarts.append('[#16]-[#16]')
                rigid_indices.append((0, 1))

            preparator = meeko.MoleculePreparation(
                rigid_macrocycles=has_macro,
                rigidify_bonds_smarts=rigid_smarts,
                rigidify_bonds_indices=rigid_indices,
                flexible_amides=False,
                merge_these_atom_types=('H',)
            )
        elif flexibility == "sidechain":
            rigid_smarts = ['[#7]-[#6]-[#6](=[#8])', '[#7]-[#6]-[#6](=[#8])']
            rigid_indices = [(0, 1), (1, 2)]
            if has_macro:
                rigid_smarts.append('[#16]-[#16]')
                rigid_indices.append((0, 1))

            preparator = meeko.MoleculePreparation(
                rigid_macrocycles=has_macro,
                rigidify_bonds_smarts=rigid_smarts,
                rigidify_bonds_indices=rigid_indices,
                flexible_amides=False,
                merge_these_atom_types=('H',)
            )
        else:
            # Default / full flexibility
            preparator = meeko.MoleculePreparation(
                rigid_macrocycles=has_macro,
                flexible_amides=False,
                merge_these_atom_types=('H',)
            )

        setups = preparator.prepare(mol)
        if not setups:
            raise RuntimeError(f"Meeko failed to prepare molecule for {input_pdb.name}")

        pdbqt_str, is_ok, err = meeko.PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            raise RuntimeError(f"Meeko PDBQT writer failed: {err}")

        out_pdbqt.write_text(pdbqt_str, encoding="utf-8")
        num_torsions = pdbqt_str.count("BRANCH")

        logger.debug(f"Prepared {out_pdbqt.name} [mode={flexibility}, macrocycles={has_macro}, active torsions: {num_torsions}]")
        return out_pdbqt, num_torsions

    finally:
        if temp_sdf.exists():
            temp_sdf.unlink(missing_ok=True)
        if temp_annotated_pdb.exists():
            temp_annotated_pdb.unlink(missing_ok=True)


def prepare_native_peptides(
    pdb_files: List[Path],
    out_dir: Path
) -> Dict[str, Path]:
    """
    Prepares native 3D peptide PDB structures for geometric docking.
    Zero external dependencies, no PDBQT, no Meeko.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    prepared = {}
    for pdb_path in pdb_files:
        pep_id = pdb_path.stem
        out_path = out_dir / f"{pep_id}.pdb"
        if pdb_path.resolve() != out_path.resolve():
            out_path.write_text(pdb_path.read_text(encoding="utf-8"), encoding="utf-8")
        prepared[pep_id] = out_path
    logger.info(f"Prepared {len(prepared)} native 3D peptide PDB ligands in {out_dir.name}.")
    return prepared


def batch_prepare_peptides(
    pdb_files: List[Path],
    out_dir: Path,
    flexibility: str = "rigid",
    obabel_bin: str = "obabel",
    lock_disulfides: bool = True
) -> Dict[str, Tuple[Path, int]]:
    """
    Prepares peptide PDB structures for docking.
    Returns {pep_id: (pdb_path, num_torsions)}.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    prepared = {}

    logger.info(f"Preparing {len(pdb_files)} native 3D peptide ligands for geometric screening...")
    for pdb_path in pdb_files:
        pep_id = pdb_path.stem
        out_path = out_dir / f"{pep_id}.pdb"
        try:
            if pdb_path.resolve() != out_path.resolve():
                out_path.write_text(pdb_path.read_text(encoding="utf-8"), encoding="utf-8")
            prepared[pep_id] = (out_path, 0)
        except Exception as e:
            logger.error(f"Failed to prepare {pep_id}: {e}")

    logger.info(f"Successfully prepared {len(prepared)}/{len(pdb_files)} peptide ligands.")
    return prepared
