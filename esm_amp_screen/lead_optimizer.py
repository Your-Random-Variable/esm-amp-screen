"""
In Silico Lead Optimization & Mutation Scanning Engine.
Provides:
1. Interface force-field relaxation (OpenBabel UFF / MMFF94 with fixed receptor atoms).
2. In silico Alanine Scanning to identify binding hotspots vs. tolerant positions.
3. Rational single-point mutation scanning across 20 amino acids to engineer
   predicted affinity gains of -0.8 to -1.5 kcal/mol (5x to 10x Kd improvements).
4. Automated biophysical mechanism attribution (salt bridges, pocket filling, H-bonds).
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gemmi
import numpy as np
try:
    from openbabel import openbabel as ob
    _HAS_OPENBABEL = True
except ImportError:
    _HAS_OPENBABEL = False

import PeptideBuilder

from esm_amp_screen.utils import logger, ensure_dir, clean_sequence
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import (
    analyze_complex_contacts,
    analyze_cocomplex_contacts,
    calculate_conformational_entropy_penalty,
    estimate_solution_affinity,
    calculate_net_charge,
    calculate_mean_hydrophobicity,
    calculate_hydrophobic_moment,
    predict_hemolytic_risk_and_selectivity
)

# Standard amino acid mappings
AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
AA_3TO1 = {v: k for k, v in AA_1TO3.items()}

# Rational library of candidate substitutions for affinity maturation
RATIONAL_MUTATION_LIBRARY = [
    'R', 'K',  # Cationic / salt-bridge anchors
    'D', 'E',  # Anionic / salt-bridge anchors
    'W', 'F', 'Y', 'L', 'I', 'V',  # Hydrophobic pocket fillers / aromatic stacking
    'Q', 'N', 'S', 'T', 'H',       # Polar H-bond anchors
    'P', 'A'                       # Structural / conformational pre-organization
]

# Empirical heuristic scaling factor converting delta HADDOCK score to an estimated
# relative delta delta G (kcal/mol) for rank-ordering candidate mutations.
# Note: delta delta G and estimated Kd fold-changes computed from this factor represent
# heuristic ranking scores for prioritizing design candidates, not experimentally calibrated
# rigorous thermodynamic free energies.
HADDOCK_TO_DG: float = 0.15



def kabsch_fit(p_src: np.ndarray, p_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes optimal rotation matrix R and translation vector t
    to superimpose 3D coordinate set p_src onto p_dst using Kabsch algorithm.
    """
    c_src = np.mean(p_src, axis=0)
    c_dst = np.mean(p_dst, axis=0)
    h = (p_src - c_src).T @ (p_dst - c_dst)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = c_dst - c_src @ r.T
    return r, t


def relax_complex_interface(
    complex_pdb: Path,
    out_pdb: Optional[Path] = None,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
    steps: int = 50,
    forcefield: str = "UFF"
) -> Path:
    """
    Relaxes interface atomic coordinates using force-field minimization with frozen
    receptor atoms. Relieves sub-angstrom steric clashes, optimizes H-bond geometry,
    and eliminates artificial grid strains without altering the global binding mode.
    """
    complex_pdb = Path(complex_pdb)
    if not out_pdb:
        out_pdb = complex_pdb.with_name(f"{complex_pdb.stem}_relaxed.pdb")
    out_pdb = Path(out_pdb)
    ensure_dir(out_pdb.parent)

    if not _HAS_OPENBABEL:
        logger.warning("OpenBabel Python module not available for force-field relaxation. Preserving original coordinates.")
        if complex_pdb.resolve() != out_pdb.resolve():
            out_pdb.write_text(complex_pdb.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return out_pdb

    try:
        mol = ob.OBMol()
        conv = ob.OBConversion()
        conv.SetInFormat("pdb")
        conv.SetOutFormat("pdb")
        if not conv.ReadFile(mol, str(complex_pdb)):
            logger.warning(f"OpenBabel could not parse {complex_pdb}. Skipping relaxation.")
            if complex_pdb.resolve() != out_pdb.resolve():
                out_pdb.write_text(complex_pdb.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            return out_pdb

        ff = ob.OBForceField.FindForceField(forcefield)
        if ff is None or not ff.Setup(mol):
            # Fallback to UFF if MMFF94 fails on metals or non-standard valences
            ff = ob.OBForceField.FindForceField("UFF")
            if ff is None or not ff.Setup(mol):
                logger.warning("Failed to setup force field for complex relaxation.")
                if complex_pdb.resolve() != out_pdb.resolve():
                    out_pdb.write_text(complex_pdb.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                return out_pdb

        # Freeze all receptor atoms
        constraints = ob.OBFFConstraints()
        rec_frozen = 0
        for atom in ob.OBMolAtomIter(mol):
            res = atom.GetResidue()
            if res and res.GetChain() == receptor_chain:
                constraints.AddAtomConstraint(atom.GetIdx())
                rec_frozen += 1

        ff.SetConstraints(constraints)
        ff.SteepestDescent(steps)
        ff.GetCoordinates(mol)
        conv.WriteFile(mol, str(out_pdb))
        logger.debug(f"Relaxed complex {complex_pdb.name}: froze {rec_frozen} receptor atoms, ran {steps} {forcefield} steps.")
        return out_pdb
    except Exception as e:
        logger.warning(f"Error during interface relaxation ({e}). Using unrelaxed complex.")
        if complex_pdb.resolve() != out_pdb.resolve():
            out_pdb.write_text(complex_pdb.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return out_pdb


def mutate_peptide_residue(
    st: gemmi.Structure,
    chain_id: str,
    res_idx: int,
    target_aa: str
) -> gemmi.Structure:
    """
    Mutates residue at 0-based index `res_idx` in chain `chain_id` to `target_aa` (1-letter).
    Preserves exact backbone N, CA, C coordinates and aligns ideal sidechain geometry via Kabsch.
    """
    target_aa = target_aa.upper()
    if target_aa not in AA_1TO3:
        raise ValueError(f"Unknown amino acid code '{target_aa}'")

    model = st[0]
    target_chain = None
    for c in model:
        if c.name == chain_id:
            target_chain = c
            break
    if not target_chain:
        raise ValueError(f"Chain {chain_id} not found in structure")

    if res_idx < 0 or res_idx >= len(target_chain):
        raise IndexError(f"Residue index {res_idx} out of range (chain length {len(target_chain)})")

    res = target_chain[res_idx]
    three_letter = AA_1TO3[target_aa]

    # Fast path for Alanine mutation
    if target_aa == "A":
        res.trim_to_alanine()
        res.name = "ALA"
        return st

    # Extract backbone coordinates
    bb_atoms = {"N": None, "CA": None, "C": None, "O": None}
    for a in res:
        if a.name in bb_atoms:
            bb_atoms[a.name] = np.array([a.pos.x, a.pos.y, a.pos.z])

    if bb_atoms["N"] is None or bb_atoms["CA"] is None or bb_atoms["C"] is None:
        raise ValueError(f"Residue {res.name}{res.seqid.num} missing backbone N, CA, or C atoms")

    p_dst = np.array([bb_atoms["N"], bb_atoms["CA"], bb_atoms["C"]])

    # Generate ideal residue geometry
    geo = PeptideBuilder.Geometry.geometry(target_aa)
    ideal_struct = PeptideBuilder.initialize_res(geo)

    ideal_atoms = []
    ideal_bb = {}
    for a in ideal_struct.get_atoms():
        coords = a.get_coord()
        ideal_atoms.append((a.name, a.element, coords))
        if a.name in ("N", "CA", "C"):
            ideal_bb[a.name] = coords

    p_src = np.array([ideal_bb["N"], ideal_bb["CA"], ideal_bb["C"]])
    r, t = kabsch_fit(p_src, p_dst)

    # Remove old sidechain atoms (keep N, CA, C, O)
    atoms_to_remove = [a.name for a in res if a.name not in ("N", "CA", "C", "O")]
    for name in atoms_to_remove:
        res.remove_atom(name, '\0')

    res.name = three_letter

    # Add new sidechain atoms
    for name, elem, coords in ideal_atoms:
        if elem == "H" or name in ("N", "CA", "C", "O"):
            continue
        new_pos = coords @ r.T + t
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(elem)
        atom.pos = gemmi.Position(new_pos[0], new_pos[1], new_pos[2])
        res.add_atom(atom)

    return st


class LeadOptimizer:
    """
    Comprehensive In Silico Lead Optimization and Affinity Maturation Engine.
    """

    def __init__(
        self,
        complex_pdb: Path,
        receptor_chain: str = "A",
        peptide_chain: str = "B",
        out_dir: Optional[Path] = None,
        pocket_residues: Optional[List[str]] = None,
        threads: int = 1,
        baseline_dg: Optional[float] = None
    ):
        self.complex_pdb = Path(complex_pdb)
        if not self.complex_pdb.exists():
            raise FileNotFoundError(f"Complex PDB not found: {self.complex_pdb}")

        self.receptor_chain = receptor_chain
        self.peptide_chain = peptide_chain
        self.out_dir = Path(out_dir or self.complex_pdb.parent / "lead_optimization")
        ensure_dir(self.out_dir)
        self.pocket_residues = pocket_residues
        self.threads = max(1, threads)

        # Parse peptide sequence and residue numbers
        self.st = gemmi.read_structure(str(self.complex_pdb))
        self.pep_chain = None
        for c in self.st[0]:
            if c.name == self.peptide_chain:
                self.pep_chain = c
                break

        if not self.pep_chain or len(self.pep_chain) == 0:
            raise ValueError(f"Peptide chain '{self.peptide_chain}' not found in {self.complex_pdb}")

        self.wt_residues: List[Tuple[int, str, int]] = []  # (res_idx, 1-letter, seqid_num)
        for idx, res in enumerate(self.pep_chain):
            one_letter = AA_3TO1.get(res.name.upper(), "X")
            self.wt_residues.append((idx, one_letter, res.seqid.num))

        self.wt_sequence = "".join(r[1] for r in self.wt_residues if r[1] != "X")
        logger.info(f"Loaded complex '{self.complex_pdb.name}' with peptide chain {self.peptide_chain}: '{self.wt_sequence}' ({len(self.wt_residues)} aa).")

        # Baseline evaluation
        self.baseline_relaxed_pdb = self.out_dir / f"baseline_{self.complex_pdb.stem}_relaxed.pdb"
        relax_complex_interface(
            self.complex_pdb,
            self.baseline_relaxed_pdb,
            receptor_chain=self.receptor_chain,
            peptide_chain=self.peptide_chain
        )

        self.baseline_physics = calculate_haddock_physics_score(
            self.baseline_relaxed_pdb,
            pocket_residues=self.pocket_residues,
            receptor_chain=self.receptor_chain,
            peptide_chain=self.peptide_chain
        )
        self.baseline_contacts = analyze_cocomplex_contacts(
            self.baseline_relaxed_pdb,
            receptor_chain=self.receptor_chain,
            peptide_chain=self.peptide_chain
        )
        # Baseline free energy: use actual complex docking delta G if provided,
        # otherwise derive from HADDOCK physical score using consistent HADDOCK_TO_DG scaling
        if baseline_dg is not None:
            self.baseline_dg = round(float(baseline_dg), 2)
        else:
            self.baseline_dg = round(HADDOCK_TO_DG * self.baseline_physics["haddock_score"], 2)

        self.baseline_sol = estimate_solution_affinity(
            affinity_kcal_mol=self.baseline_dg,
            sequence=self.wt_sequence
        )
        logger.info(f"Baseline Score: HADDOCK={self.baseline_physics['haddock_score']:.1f}, BSA={self.baseline_physics['bsa']:.1f} Å², Sol. Kd={self.baseline_sol['solution_kd']}")

    def run_alanine_scan(self, threads: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs in silico Alanine Scanning across all positions in the peptide.
        Classifies each position as:
        - 'HOTSPOT (Critical)' (DeltaDelta G >= +1.2 kcal/mol or DeltaHADDOCK >= +4.0)
        - 'CONTRIBUTOR' (+0.4 <= DeltaDelta G < +1.2 kcal/mol)
        - 'TOLERANT (Lead Opt Candidate)' (DeltaDelta G < +0.4 kcal/mol)
        """
        active_threads = threads or self.threads
        logger.info(f"Starting In Silico Alanine Scan for {len(self.wt_residues)} residues (threads={active_threads})...")

        def _scan_single(item):
            idx, wt_aa, res_num = item
            target_mut = "A" if wt_aa != "A" else "G"
            mut_id = f"{wt_aa}{res_num}{target_mut}"

            st_mut = self.st.clone()
            try:
                mutate_peptide_residue(st_mut, self.peptide_chain, idx, target_mut)
                unrel_path = self.out_dir / f"ala_scan_{mut_id}.unrel.pdb"
                st_mut.write_minimal_pdb(str(unrel_path))

                rel_path = self.out_dir / f"ala_scan_{mut_id}.pdb"
                relax_complex_interface(
                    unrel_path,
                    rel_path,
                    receptor_chain=self.receptor_chain,
                    peptide_chain=self.peptide_chain,
                    steps=30
                )
                if unrel_path.exists():
                    unrel_path.unlink(missing_ok=True)

                phys = calculate_haddock_physics_score(
                    rel_path,
                    pocket_residues=self.pocket_residues,
                    receptor_chain=self.receptor_chain,
                    peptide_chain=self.peptide_chain
                )
                contacts = analyze_cocomplex_contacts(
                    rel_path,
                    receptor_chain=self.receptor_chain,
                    peptide_chain=self.peptide_chain
                )

                delta_haddock = round(phys["haddock_score"] - self.baseline_physics["haddock_score"], 2)
                # In alanine scanning: positive delta means binding got WORSE without the sidechain (it was a hotspot)
                delta_delta_g = round(HADDOCK_TO_DG * delta_haddock, 2)
                delta_bsa = round(phys["bsa"] - self.baseline_physics["bsa"], 1)

                if delta_delta_g >= 1.2 or delta_haddock >= 4.0:
                    role = "HOTSPOT (Critical)"
                elif delta_delta_g >= 0.4:
                    role = "CONTRIBUTOR"
                else:
                    role = "TOLERANT (Lead Opt Candidate)"

                return {
                    "position": res_num,
                    "wt_res": wt_aa,
                    "mut_res": target_mut,
                    "mut_id": mut_id,
                    "haddock_score": phys["haddock_score"],
                    "delta_haddock": delta_haddock,
                    "delta_delta_g": delta_delta_g,
                    "bsa": phys["bsa"],
                    "delta_bsa": delta_bsa,
                    "hbonds": contacts["hbonds"],
                    "salt_bridges": contacts["salt_bridges"],
                    "classification": role
                }
            except Exception as e:
                logger.warning(f"Alanine scan failed for {mut_id}: {e}")
                return None

        results = []
        if active_threads > 1 and len(self.wt_residues) > 1:
            with ThreadPoolExecutor(max_workers=min(active_threads, len(self.wt_residues))) as ex:
                raw = list(ex.map(_scan_single, self.wt_residues))
                results = [r for r in raw if r is not None]
        else:
            for item in self.wt_residues:
                r = _scan_single(item)
                if r is not None:
                    results.append(r)

        results.sort(key=lambda x: x["position"])
        return results

    def run_lead_optimization(
        self,
        candidate_positions: Optional[List[int]] = None,
        candidate_amino_acids: Optional[List[str]] = None,
        top_n: int = 15,
        threads: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates single-point mutations to identify variants that boost affinity
        by the target -0.8 to -1.5 kcal/mol (5x to 10x Kd gain).
        """
        active_threads = threads or self.threads
        mut_aas = candidate_amino_acids or RATIONAL_MUTATION_LIBRARY

        # Determine target positions
        if candidate_positions:
            target_res_tuples = [r for r in self.wt_residues if r[2] in candidate_positions]
        else:
            # If not specified, first run alanine scan to prioritize tolerant and interface residues
            ala_res = self.run_alanine_scan(threads=active_threads)
            tolerant_nums = {r["position"] for r in ala_res if r["classification"] != "HOTSPOT (Critical)"}
            target_res_tuples = [r for r in self.wt_residues if r[2] in tolerant_nums]
            if not target_res_tuples:
                target_res_tuples = self.wt_residues

        variants_dir = self.out_dir / "optimized_variants"
        ensure_dir(variants_dir)

        r_const = 1.9872e-3
        t_kelvin = 298.15

        task_list = []
        for idx, wt_aa, res_num in target_res_tuples:
            for mut_aa in mut_aas:
                if mut_aa != wt_aa:
                    task_list.append((idx, wt_aa, res_num, mut_aa))

        logger.info(f"Screening {len(task_list)} lead optimization mutations across {len(target_res_tuples)} positions (threads={active_threads})...")

        def _eval_single_mut(task):
            idx, wt_aa, res_num, mut_aa = task
            mut_id = f"{wt_aa}{res_num}{mut_aa}"

            mut_seq_list = list(self.wt_sequence)
            if idx < len(mut_seq_list):
                mut_seq_list[idx] = mut_aa
            mut_seq = "".join(mut_seq_list)

            st_mut = self.st.clone()
            try:
                mutate_peptide_residue(st_mut, self.peptide_chain, idx, mut_aa)
                unrel_path = variants_dir / f"mut_{mut_id}.unrel.pdb"
                st_mut.write_minimal_pdb(str(unrel_path))

                rel_path = variants_dir / f"complex_{mut_id}.pdb"
                relax_complex_interface(
                    unrel_path,
                    rel_path,
                    receptor_chain=self.receptor_chain,
                    peptide_chain=self.peptide_chain,
                    steps=40
                )
                if unrel_path.exists():
                    unrel_path.unlink(missing_ok=True)

                phys = calculate_haddock_physics_score(
                    rel_path,
                    pocket_residues=self.pocket_residues,
                    receptor_chain=self.receptor_chain,
                    peptide_chain=self.peptide_chain
                )
                contacts = analyze_cocomplex_contacts(
                    rel_path,
                    receptor_chain=self.receptor_chain,
                    peptide_chain=self.peptide_chain
                )

                delta_haddock = round(phys["haddock_score"] - self.baseline_physics["haddock_score"], 2)
                delta_evdw = round(phys["e_vdw"] - self.baseline_physics["e_vdw"], 2)
                delta_eelec = round(phys["e_elec"] - self.baseline_physics["e_elec"], 2)
                delta_edesolv = round(phys["e_desolv"] - self.baseline_physics["e_desolv"], 2)
                delta_bsa = round(phys["bsa"] - self.baseline_physics["bsa"], 1)

                # Predicted DeltaDelta G (kcal/mol): negative = favorable affinity boost
                delta_delta_g = round(HADDOCK_TO_DG * delta_haddock, 2)
                pred_dg = round(self.baseline_dg + delta_delta_g, 2)

                # Solution Kd with entropy penalty
                sol_info = estimate_solution_affinity(
                    affinity_kcal_mol=pred_dg,
                    sequence=mut_seq
                )

                # Kd Fold Improvement: exp(-DeltaDelta G / RT)
                fold_boost = round(math.exp(-delta_delta_g / (r_const * t_kelvin)), 1) if delta_delta_g < 0 else round(1.0 / math.exp(delta_delta_g / (r_const * t_kelvin)), 2)

                # Determine biophysical mechanism of gain
                mechanisms = []
                if contacts["salt_bridges"] > self.baseline_contacts["salt_bridges"]:
                    mechanisms.append(f"Formed {contacts['salt_bridges'] - self.baseline_contacts['salt_bridges']} new salt bridge(s)")
                if delta_edesolv <= -1.5 and delta_bsa >= 10.0:
                    mechanisms.append(f"Hydrophobic pocket burial (+{delta_bsa:.0f} Å² BSA)")
                if contacts["hbonds"] > self.baseline_contacts["hbonds"]:
                    mechanisms.append(f"Additional H-bond (+{contacts['hbonds'] - self.baseline_contacts['hbonds']})")
                if delta_eelec <= -3.0 and not any("salt" in m for m in mechanisms):
                    mechanisms.append(f"Electrostatic attraction (ΔEelec={delta_eelec:.1f})")
                if not mechanisms:
                    mechanisms.append("Interface van der Waals packing" if delta_evdw < 0 else "Neutral substitution")

                # Toxicity & Selectivity Check
                net_q = calculate_net_charge(mut_seq)
                h_mom = calculate_hydrophobic_moment(mut_seq)
                mean_h = calculate_mean_hydrophobicity(mut_seq)
                tox = predict_hemolytic_risk_and_selectivity(net_q, mean_h, h_mom)

                return {
                    "mutation": mut_id,
                    "position": res_num,
                    "wt_res": wt_aa,
                    "mut_res": mut_aa,
                    "mut_sequence": mut_seq,
                    "haddock_score": phys["haddock_score"],
                    "delta_haddock": delta_haddock,
                    "pred_delta_g": pred_dg,
                    "delta_delta_g": delta_delta_g,
                    "solution_kd": sol_info["solution_kd"],
                    "kd_fold_improvement": fold_boost,
                    "bsa": phys["bsa"],
                    "delta_bsa": delta_bsa,
                    "salt_bridges": contacts["salt_bridges"],
                    "hbonds": contacts["hbonds"],
                    "e_vdw": phys["e_vdw"],
                    "e_elec": phys["e_elec"],
                    "e_desolv": phys["e_desolv"],
                    "hemolytic_risk": tox["hemolytic_risk"],
                    "selectivity": tox["therapeutic_selectivity"],
                    "mechanism": "; ".join(mechanisms),
                    "complex_pdb": str(rel_path)
                }
            except Exception as e:
                logger.debug(f"Mutation {mut_id} evaluation error: {e}")
                return None

        variants = []
        if active_threads > 1 and len(task_list) > 1:
            with ThreadPoolExecutor(max_workers=min(active_threads, len(task_list))) as ex:
                raw = list(ex.map(_eval_single_mut, task_list))
                variants = [v for v in raw if v is not None]
        else:
            for task in task_list:
                v = _eval_single_mut(task)
                if v is not None:
                    variants.append(v)

        variants.sort(key=lambda x: x["delta_delta_g"])
        return variants[:top_n]
