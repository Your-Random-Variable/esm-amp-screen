"""
HADDOCK-Style Molecular Physics Scoring Engine.
Calculates explicit intermolecular physical forces according to published standards:
- 6-12 Lennard-Jones van der Waals with Lorentz-Berthelot mixing rules (E_vdw)
- Coulombic electrostatics with distance-dependent dielectric epsilon(r) = 4r (E_elec)
- Atomic desolvation free energy via Eisenberg & McLachlan atomic solvation parameters (E_desolv)
- Solvent-Accessible & Buried Surface Area (BSA) via numerical Shrake-Rupley
- Ambiguous Interaction Restraints (E_AIR) to target active site residues
- Standard Published HADDOCK 2.4 Score: 1.0*E_vdw + 0.2*E_elec + 1.0*E_desolv + 0.1*E_AIR
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import gemmi

from esm_amp_screen.utils import logger

# Van der Waals parameters (sigma in Angstroms, epsilon in kcal/mol) — OPLS-AA (Jorgensen et al. JACS 1996)
# Sigma is the Lennard-Jones collision diameter (zero-energy distance).
VDW_PARAMS: Dict[str, Tuple[float, float]] = {
    "C":  (3.50, 0.066),   # Aliphatic carbon (CT)
    "N":  (3.25, 0.170),   # Amide / amino nitrogen
    "O":  (2.96, 0.210),   # Carbonyl / carboxyl oxygen
    "S":  (3.55, 0.250),   # Thiol / thioether sulfur
    "H":  (2.50, 0.030),   # Aliphatic hydrogen (OPLS-AA standard: 2.50 A, 0.030 kcal/mol)
    "P":  (3.74, 0.200),   # Phosphorus
    "F":  (3.12, 0.061),   # Fluorine
    "CL": (3.65, 0.265),   # Chlorine
    "BR": (3.83, 0.320),   # Bromine
}

# Eisenberg & McLachlan (Nature 1986) Atomic Solvation Parameters (ASP) in kcal / (mol * A^2)
# Sign convention for transfer from water to protein interior (burial):
# Favorable to bury hydrophobic surface (C, S) -> negative free energy (dG < 0)
# Unfavorable to bury polar surface (N, O) without water -> positive free energy (dG > 0)
SOLVATION_PARAMS: Dict[str, float] = {
    "C":  -0.016,   # Favorable to bury nonpolar carbon
    "N":  +0.006,   # Unfavorable desolvation penalty to bury polar nitrogen
    "O":  +0.006,   # Unfavorable desolvation penalty to bury polar oxygen
    "S":  -0.021,   # Favorable to bury sulfur
    "H":   0.000,   # Nonpolar hydrogens accounted for implicitly
    "P":  +0.012,
}

# AMBER ff14SB Standard Partial Charges (elementary charge units e) for all-atom structures
# References: Maier et al., J. Chem. Theory Comput. 2015; Hornak et al., Proteins 2006
ALL_ATOM_CHARGES: Dict[Tuple[str, str], float] = {
    # --- Arginine guanidinium (+1 net charge) ---
    ("ARG", "NE"):  -0.5295, ("ARG", "HE"):   0.3456,
    ("ARG", "CZ"):   0.7341,
    ("ARG", "NH1"): -0.3549, ("ARG", "HH11"): 0.3593, ("ARG", "HH12"): 0.3593,
    ("ARG", "NH2"): -0.3549, ("ARG", "HH21"): 0.3593, ("ARG", "HH22"): 0.3593,
    ("ARG", "CD"):   0.2065, ("ARG", "CG"):   0.0500,

    # --- Lysine ammonium (+1 net charge) ---
    ("LYS", "NZ"):  -0.3854,
    ("LYS", "HZ1"):  0.3300, ("LYS", "HZ2"):  0.3300, ("LYS", "HZ3"):  0.3300,
    ("LYS", "CE"):  -0.0143,

    # --- Aspartate carboxylate (-1 net charge) ---
    ("ASP", "CG"):   0.7994, ("ASP", "OD1"): -0.8014, ("ASP", "OD2"): -0.8014,

    # --- Glutamate carboxylate (-1 net charge) ---
    ("GLU", "CD"):   0.8054, ("GLU", "OE1"): -0.8188, ("GLU", "OE2"): -0.8188,

    # --- Histidine tautomers at pH 7.4 ---
    # HID (neutral, ND1 protonated)
    ("HID", "ND1"): -0.3811, ("HID", "HD1"):  0.3648,
    ("HID", "NE2"): -0.5727, ("HID", "CE1"):  0.2050,
    # HIE (neutral, NE2 protonated)
    ("HIE", "ND1"): -0.5432,
    ("HIE", "NE2"): -0.2737, ("HIE", "HE2"):  0.3648, ("HIE", "CE1"):  0.2050,
    # HIP (cationic +1, doubly protonated)
    ("HIP", "ND1"): -0.2500, ("HIP", "HD1"):  0.3600,
    ("HIP", "NE2"): -0.2500, ("HIP", "HE2"):  0.3600, ("HIP", "CE1"):  0.2500,
    # Standard HIS fallback (weighted 90% neutral HID/HIE, ~10% protonated)
    ("HIS", "ND1"): -0.4200, ("HIS", "NE2"): -0.4500, ("HIS", "CE1"):  0.2050,

    # --- Tyrosine ---
    ("TYR", "OH"):  -0.5580, ("TYR", "HH"):   0.3990, ("TYR", "CZ"):   0.3220,

    # --- Cysteine ---
    ("CYS", "SG"):  -0.3120, ("CYS", "HG"):   0.1930,

    # --- Universal Peptide Backbone (ALL pseudo-residue) ---
    ("ALL", "N"):   -0.4157,
    ("ALL", "H"):    0.2719, ("ALL", "HN"):   0.2719,
    ("ALL", "CA"):   0.0337,
    ("ALL", "C"):    0.5973,
    ("ALL", "O"):   -0.5679,
}

# Coarse-grained United-Atom / Implicit-Hydrogen Partial Charges for heavy-atom PDB structures.
# Assigned charges collapse polar hydrogen charges onto their parent heavy atoms:
#   ARG is net +1.00 (guanidinium: CZ, NH1, NH2, NE)
#   LYS is net +1.00 (ammonium: NZ)
#   ASP is net -1.00 (carboxylate: CG, OD1, OD2)
#   GLU is net -1.00 (carboxylate: CD, OE1, OE2)
#   HIP is net +1.00 (cationic imidazolium: ND1, NE2)
#   Backbone peptide groups and neutral sidechains sum to 0.00.
# Note: This is an empirical united-atom scoring heuristic providing rank-ordering
# guidance for peptide screening, rather than an all-atom solvent-screened dipole model.
HEAVY_ATOM_CHARGES: Dict[Tuple[str, str], float] = {
    # --- Arginine guanidinium (+1 net formal charge) ---
    ("ARG", "CZ"):   0.6400,
    ("ARG", "NH1"):  0.3600,
    ("ARG", "NH2"):  0.3600,
    ("ARG", "NE"):  -0.3600,

    # --- Lysine ammonium (+1 net formal charge) ---
    ("LYS", "NZ"):   1.0000,

    # --- Aspartate carboxylate (-1 net formal charge) ---
    ("ASP", "CG"):   0.6000,
    ("ASP", "OD1"): -0.8000,
    ("ASP", "OD2"): -0.8000,

    # --- Glutamate carboxylate (-1 net formal charge) ---
    ("GLU", "CD"):   0.6000,
    ("GLU", "OE1"): -0.8000,
    ("GLU", "OE2"): -0.8000,

    # --- Histidine (neutral at pH 7.4; HIP cationic +1) ---
    ("HIP", "ND1"):  0.5000,
    ("HIP", "NE2"):  0.5000,
    ("HID", "ND1"):  0.0000,
    ("HIE", "NE2"):  0.0000,
    ("HIS", "ND1"):  0.0000,
    ("HIS", "NE2"):  0.0000,

    # --- Universal Peptide Backbone (neutral dipole: sum = 0.00) ---
    ("ALL", "C"):    0.5500,
    ("ALL", "O"):   -0.5500,
    ("ALL", "OXT"): -0.5500,
    ("ALL", "N"):    0.0000,
    ("ALL", "CA"):   0.0000,
}

# Alias for backwards compatibility
PARTIAL_CHARGES: Dict[Tuple[str, str], float] = ALL_ATOM_CHARGES


def get_residue_formal_charge(res_name: str, has_hydrogens: bool = False) -> float:
    """Computes the net formal charge of a standard amino acid residue (pH 7.4)."""
    if not has_hydrogens:
        return round(float(sum(v for k, v in HEAVY_ATOM_CHARGES.items() if k[0] == res_name)), 2)
    _STANDARD_FORMAL_CHARGES = {
        "ARG": 1.0, "LYS": 1.0, "HIP": 1.0,
        "ASP": -1.0, "GLU": -1.0,
    }
    return _STANDARD_FORMAL_CHARGES.get(res_name, 0.0)


# Bondi (1964) van der Waals radii for Solvent Accessible Surface Area (SASA)
VDW_RADII: Dict[str, float] = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85
}


def _generate_fibonacci_sphere(n: int = 96) -> np.ndarray:
    """Generates quasi-uniform points on a unit sphere using the Fibonacci lattice."""
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
    theta = 2.0 * np.pi * i / golden_ratio
    return np.column_stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ])


SPHERE_POINTS = _generate_fibonacci_sphere(96)


def compute_shrake_rupley_sasa(
    coords: np.ndarray,
    elements: List[str],
    probe_radius: float = 1.4
) -> Tuple[float, np.ndarray]:
    """
    Computes Solvent Accessible Surface Area (SASA) using the Shrake-Rupley (1973) algorithm.
    Returns (total_sasa, per_atom_sasa).
    """
    n_atoms = len(coords)
    if n_atoms == 0:
        return 0.0, np.zeros(0)

    radii = np.array([VDW_RADII.get(e, 1.70) for e in elements])
    tree = cKDTree(coords)
    atom_sasa = np.zeros(n_atoms)

    max_atom_r = float(np.max(radii))
    max_neighbor_r = max_atom_r + probe_radius

    for i, (center, r_i) in enumerate(zip(coords, radii)):
        eff_radius = r_i + probe_radius
        test_pts = center + SPHERE_POINTS * eff_radius

        search_r = eff_radius + max_neighbor_r
        neighbors = tree.query_ball_point(center, search_r)
        if i in neighbors:
            neighbors.remove(i)

        if not neighbors:
            atom_sasa[i] = 4.0 * np.pi * (eff_radius ** 2)
            continue

        neigh_coords = coords[neighbors]
        neigh_block_radii = radii[neighbors] + probe_radius

        # Check if test points fall within any neighboring atom blocking sphere
        dists = np.linalg.norm(test_pts[:, None, :] - neigh_coords[None, :, :], axis=2)
        buried = np.any(dists < neigh_block_radii[None, :], axis=1)
        exposed_fraction = 1.0 - (np.sum(buried) / len(test_pts))
        atom_sasa[i] = exposed_fraction * 4.0 * np.pi * (eff_radius ** 2)

    return float(np.sum(atom_sasa)), atom_sasa


def calculate_haddock_physics_score(
    complex_pdb: Path,
    pocket_residues: Optional[List[str]] = None,
    receptor_chain: str = "A",
    peptide_chain: str = "B"
) -> Dict[str, Any]:
    """
    Computes HADDOCK-style explicit intermolecular physical energy terms:
    - E_vdw: 6-12 Lennard-Jones van der Waals with Lorentz-Berthelot arithmetic mixing
    - E_elec: Coulombic electrostatics with distance-dependent dielectric epsilon(r) = 4r
    - E_desolv: Atomic desolvation free energy (Eisenberg & McLachlan per-atom ASP model)
    - BSA: Buried Surface Area (A^2)
    - E_AIR: Ambiguous Interaction Restraint penalty (residue-level, bounded)
    - haddock_score: Calibrated composite score (published HADDOCK 2.4 weights)
    """
    complex_pdb = Path(complex_pdb)
    if not complex_pdb.exists():
        raise FileNotFoundError(f"Complex PDB not found: {complex_pdb}")

    st = gemmi.read_structure(str(complex_pdb))

    rec_atoms = []
    pep_atoms = []
    all_atoms = []

    # Universal Coarse-Grained United-Atom Scoring:
    # Explicit hydrogens are ignored in distance calculations and charge assignments,
    # as non-polar hydrogens are implicitly parameterized in heavy-atom VDW and united-atom charges.
    for model in st:
        for chain in model:
            for res in chain:
                res_atom_names = {a.name for a in res}
                for atom in res:
                    # Ignore explicit hydrogens for consistent united-atom physics
                    if atom.element.name == "H":
                        continue

                    # Unpaired carbonyl C guard: if terminal or fragment residue has C
                    # without its paired O or OXT, do not assign the +0.55 dipole charge
                    if atom.name == "C" and "O" not in res_atom_names and "OXT" not in res_atom_names:
                        q = 0.0
                    else:
                        q = HEAVY_ATOM_CHARGES.get(
                            (res.name, atom.name),
                            HEAVY_ATOM_CHARGES.get(("ALL", atom.name), 0.0)
                        )

                    info = {
                        "coord": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                        "elem": atom.element.name,
                        "res_name": res.name,
                        "atom_name": atom.name,
                        "res_num": res.seqid.num,
                        "chain": chain.name,
                        "q": q
                    }
                    all_atoms.append(info)
                    if chain.name == receptor_chain:
                        rec_atoms.append(info)
                    elif chain.name == peptide_chain:
                        pep_atoms.append(info)

    if not rec_atoms or not pep_atoms:
        logger.warning(
            f"Could not separate chains {receptor_chain} and {peptide_chain} in {complex_pdb.name}"
        )
        return {
            "e_vdw": 0.0, "e_elec": 0.0, "e_desolv": 0.0,
            "bsa": 0.0, "e_air": 0.0, "haddock_score": 0.0
        }

    r_coords = np.array([a["coord"] for a in rec_atoms])
    p_coords = np.array([a["coord"] for a in pep_atoms])
    all_coords = np.array([a["coord"] for a in all_atoms])

    r_elems = [a["elem"] for a in rec_atoms]
    p_elems = [a["elem"] for a in pep_atoms]
    all_elems = [a["elem"] for a in all_atoms]

    r_qs = np.array([a["q"] for a in rec_atoms])
    p_qs = np.array([a["q"] for a in pep_atoms])

    # 1. Buried Surface Area (BSA) & Per-Atom Desolvation (Eisenberg & McLachlan)
    sasa_rec, rec_atom_sasa_free = compute_shrake_rupley_sasa(r_coords, r_elems)
    sasa_pep, pep_atom_sasa_free = compute_shrake_rupley_sasa(p_coords, p_elems)
    sasa_cpx, cpx_atom_sasa = compute_shrake_rupley_sasa(all_coords, all_elems)
    bsa = max(0.0, (sasa_rec + sasa_pep) - sasa_cpx)

    # In cpx_atom_sasa: first len(rec_atoms) atoms are receptor, remainder are peptide
    rec_cpx_sasa = cpx_atom_sasa[:len(rec_atoms)]
    pep_cpx_sasa = cpx_atom_sasa[len(rec_atoms):]

    # Delta SASA for each atom upon complex formation
    delta_sasa_rec = np.maximum(0.0, rec_atom_sasa_free - rec_cpx_sasa)
    delta_sasa_pep = np.maximum(0.0, pep_atom_sasa_free - pep_cpx_sasa)

    # Atomic desolvation: E_desolv = sum(sigma_k * Delta_SASA_k)
    e_desolv = 0.0
    for el, d_sasa in zip(r_elems, delta_sasa_rec):
        sigma = SOLVATION_PARAMS.get(el, 0.0)
        e_desolv += sigma * float(d_sasa)
    for el, d_sasa in zip(p_elems, delta_sasa_pep):
        sigma = SOLVATION_PARAMS.get(el, 0.0)
        e_desolv += sigma * float(d_sasa)

    # 2. Intermolecular Distance Matrix & Force Fields
    dists = cdist(r_coords, p_coords)
    cutoff_mask = dists < 12.0  # 12 A interaction sphere (HADDOCK standard)

    e_vdw = 0.0
    e_elec = 0.0
    coulomb_const = 332.0636

    # Polynomial switching function parameters (r_on = 9.0 A, r_off = 12.0 A)
    # Tapers forces continuously to 0 at the cutoff boundary to prevent step discontinuities
    r_on = 9.0
    r_off = 12.0
    r_range = r_off - r_on

    for i in range(len(rec_atoms)):
        j_indices = np.where(cutoff_mask[i])[0]
        if len(j_indices) == 0:
            continue

        for j in j_indices:
            r = dists[i, j]

            # Canonical C2 quintic smoothstep switching function:
            # S(u) = 1.0 - 10.0*u^3 + 15.0*u^4 - 6.0*u^5 where u = (r - 9) / 3
            # Zero 1st and 2nd derivatives at both boundaries (r=9.0 and r=12.0)
            if r <= r_on:
                s_r = 1.0
            elif r >= r_off:
                s_r = 0.0
            else:
                u = (r - r_on) / r_range
                s_r = 1.0 - 10.0 * (u ** 3) + 15.0 * (u ** 4) - 6.0 * (u ** 5)

            # Van der Waals: 6-12 Lennard-Jones with Lorentz-Berthelot mixing rules
            # Arithmetic mean for sigma: sig_ij = (sig_i + sig_j) / 2.0
            # Geometric mean for epsilon: eps_ij = sqrt(eps_i * eps_j)
            sig_i, eps_i = VDW_PARAMS.get(r_elems[i], (3.50, 0.066))
            sig_j, eps_j = VDW_PARAMS.get(p_elems[j], (3.50, 0.066))
            sig_ij = (sig_i + sig_j) / 2.0
            eps_ij = np.sqrt(eps_i * eps_j)

            # Preserve steep repulsive wall for sub-angstrom interpenetrations.
            # Using r_eff = max(r, 1.0) ensures that any severe overlap (including r < 0.8 A)
            # receives the maximum repulsive barrier (+500.0 kcal/mol per pair) without division by zero.
            r_eff = max(r, 1.0)
            sr6 = (sig_ij / r_eff) ** 6
            vdw_ij = 4.0 * eps_ij * (sr6 ** 2 - sr6)
            # Allow steep repulsive wall up to +500.0 kcal/mol per clashing pair;
            # cap attractive well at -2.0 kcal/mol
            e_vdw += float(np.clip(vdw_ij, -2.0, 500.0)) * s_r

            # Electrostatics: Coulomb with distance-dependent dielectric epsilon(r) = 4 * r
            qi = r_qs[i]
            qj = p_qs[j]
            if abs(qi) > 1e-4 and abs(qj) > 1e-4:
                elec_ij = (coulomb_const * qi * qj) / (4.0 * (r_eff ** 2))
                e_elec += float(np.clip(elec_ij, -150.0, 150.0)) * s_r

    # 3. Ambiguous Interaction Restraints (E_AIR)
    # HADDOCK standard: evaluates whether the ligand contacts active site pocket residues.
    # Evaluated with upper bound d_U = 3.5 A inter-heavy atom distance.
    e_air = 0.0
    if pocket_residues:
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
                    target_res_keys.add((receptor_chain, int(parts[0])))
                except ValueError:
                    continue

        res_min_dists = {}
        for i, atom in enumerate(rec_atoms):
            key = (atom["chain"], atom["res_num"])
            if key in target_res_keys:
                d = float(np.min(dists[i, :]))
                if key not in res_min_dists or d < res_min_dists[key]:
                    res_min_dists[key] = d

        if res_min_dists:
            overall_min_dist = min(res_min_dists.values())
            # Restraint threshold: contacts within 3.5 A
            if overall_min_dist > 3.5:
                e_air = min(100.0, 10.0 * ((overall_min_dist - 3.5) ** 2))
            elif len(target_res_keys) <= 5:
                # User-defined specific catalytic/active site residues
                for d in res_min_dists.values():
                    if d > 3.5:
                        e_air += min(20.0, (d - 3.5) ** 2)
            else:
                # Auto-detected surface pocket: verify peptide is in contact with >= 1 residue
                n_contacts = sum(1 for d in res_min_dists.values() if d <= 3.5)
                if n_contacts < 1:
                    e_air = 25.0

    # 4. Standard Calibrated HADDOCK 2.4 Score
    # Published Formula: 1.0 * E_vdw + 0.2 * E_elec + 1.0 * E_desolv + 0.1 * E_AIR
    # Reference: de Vries et al., Nat. Protocols 2010; van Zundert et al., J. Mol. Biol. 2016
    haddock_score = (
        1.0 * e_vdw +
        0.2 * e_elec +
        1.0 * e_desolv +
        0.1 * e_air
    )

    return {
        "e_vdw": round(float(e_vdw), 2),
        "e_elec": round(float(e_elec), 2),
        "e_desolv": round(float(e_desolv), 2),
        "bsa": round(float(bsa), 1),
        "e_air": round(float(e_air), 2),
        "haddock_score": round(float(haddock_score), 2)
    }
