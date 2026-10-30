"""
AMP Biophysical Profiler & Intermolecular Contact Analyzer.
Calculates net charge, Eisenberg hydrophobic moment, Boman index,
and extracts target-ligand interface contacts (salt bridges, H-bonds, clashes).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gemmi
import numpy as np

from esm_amp_screen.utils import logger, clean_sequence

# Eisenberg Consensus Hydrophobicity Scale (normalized, kcal/mol).
# Reference: Eisenberg et al., J. Mol. Biol. 179:125-142 (1984).
EISENBERG_SCALE = {
    'A':  0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C':  0.29,
    'Q': -0.85, 'E': -0.74, 'G':  0.48, 'H': -0.40, 'I':  1.38,
    'L':  1.06, 'K': -1.50, 'M':  0.64, 'F':  1.19, 'P':  0.12,
    'S': -0.18, 'T': -0.05, 'W':  0.81, 'Y':  0.26, 'V':  1.08
}

# Boman Protein-Interaction Free Energy Scale (kcal/mol).
# Reference: Boman, J. Intern. Med. 254:197-215 (2003).
# Positive values indicate greater interaction potential.
BOMAN_SCALE = {
    'A': 0.17, 'R': 2.11, 'N': 0.42, 'D': 0.61, 'C': 0.17,
    'Q': 0.58, 'E': 0.51, 'G': 0.00, 'H': 0.17, 'I': 1.25,
    'L': 1.06, 'K': 1.81, 'M': 0.23, 'F': 1.26, 'P': 0.00,
    'S': 0.13, 'T': 0.14, 'W': 0.87, 'Y': 0.77, 'V': 0.78
}

# pKa values for ionizable groups (used in Henderson-Hasselbalch charge calculation).
# References: Lehninger Biochemistry; Pace et al. Protein Sci. 2009.
PKA_VALUES = {
    "N_term": 8.0,    # α-amino group
    "C_term": 3.1,    # α-carboxyl group
    "K":  10.5,       # Lysine ε-amino (NZ)
    "R":  12.5,       # Arginine guanidinium (NE/NH)
    "H":   6.0,       # Histidine imidazole (ND1)
    "D":   3.9,       # Aspartate β-carboxyl
    "E":   4.1,       # Glutamate γ-carboxyl
    "C":   8.3,       # Cysteine thiol (SG)
    "Y":  10.1,       # Tyrosine phenol (OH)
}

# Sets of positively and negatively charged residue names at pH 7
_POS_RESIDUES = {"ARG", "LYS", "HIS"}
_NEG_RESIDUES = {"ASP", "GLU"}

# Hydrophobic residues for contact analysis
_HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}


def calculate_net_charge(sequence: str, ph: float = 7.4) -> float:
    """
    Computes net charge of a peptide at given pH using Henderson-Hasselbalch equation.

    For a group with pKa = pKa_ref:
      - Positive group (acid form = charged): fraction_charged = 1 / (1 + 10^(pH - pKa))
      - Negative group (base form = charged): fraction_charged = 1 / (1 + 10^(pKa - pH))

    Reference: Henderson-Hasselbalch; Bjellqvist et al. Electrophoresis 1993.
    """
    seq = clean_sequence(sequence)

    # Positive contributions (protonated = charged at low pH)
    pos = (1.0 / (1.0 + 10 ** (ph - PKA_VALUES["N_term"])))       # N-terminus α-NH3+
    pos += seq.count('K') * (1.0 / (1.0 + 10 ** (ph - PKA_VALUES["K"])))
    pos += seq.count('R') * (1.0 / (1.0 + 10 ** (ph - PKA_VALUES["R"])))
    pos += seq.count('H') * (1.0 / (1.0 + 10 ** (ph - PKA_VALUES["H"])))

    # Negative contributions (deprotonated = charged at high pH)
    neg = (1.0 / (1.0 + 10 ** (PKA_VALUES["C_term"] - ph)))       # C-terminus α-COO-
    neg += seq.count('D') * (1.0 / (1.0 + 10 ** (PKA_VALUES["D"] - ph)))
    neg += seq.count('E') * (1.0 / (1.0 + 10 ** (PKA_VALUES["E"] - ph)))
    neg += seq.count('C') * (1.0 / (1.0 + 10 ** (PKA_VALUES["C"] - ph)))
    neg += seq.count('Y') * (1.0 / (1.0 + 10 ** (PKA_VALUES["Y"] - ph)))

    return round(pos - neg, 2)


def calculate_isoelectric_point(sequence: str, tolerance: float = 0.01) -> float:
    """
    Computes theoretical isoelectric point (pI) where net charge equals 0.0.
    Uses bisection search over Henderson-Hasselbalch charge profile across pH [2.0, 13.0].
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 7.0

    low_ph = 2.0
    high_ph = 13.0

    for _ in range(25):
        mid_ph = (low_ph + high_ph) / 2.0
        charge = calculate_net_charge(seq, ph=mid_ph)
        if abs(charge) < tolerance:
            return round(mid_ph, 2)
        if charge > 0:
            low_ph = mid_ph
        else:
            high_ph = mid_ph

    return round((low_ph + high_ph) / 2.0, 2)


def calculate_hydrophobic_moment(sequence: str, angle_deg: float = 100.0) -> float:
    """
    Computes the Eisenberg hydrophobic moment (μH) for a peptide.

    μH = (1/N) · √[(Σ H_i·cos(i·δ))² + (Σ H_i·sin(i·δ))²]

    where δ is the inter-residue angular increment (100° for α-helix, 160° for β-sheet),
    H_i is the Eisenberg hydrophobicity of residue i, and N is the peptide length.

    Reference: Eisenberg et al., PNAS 81:140-144 (1984).
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0

    angle_rad = math.radians(angle_deg)
    sum_cos = 0.0
    sum_sin = 0.0

    for i, aa in enumerate(seq):
        h = EISENBERG_SCALE.get(aa, 0.0)
        sum_cos += h * math.cos(i * angle_rad)
        sum_sin += h * math.sin(i * angle_rad)

    mu_h = math.sqrt(sum_cos ** 2 + sum_sin ** 2) / len(seq)
    return round(mu_h, 3)


def calculate_mean_hydrophobicity(sequence: str) -> float:
    """Computes mean Eisenberg hydrophobicity of peptide sequence."""
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    total = sum(EISENBERG_SCALE.get(aa, 0.0) for aa in seq)
    return round(total / len(seq), 3)


def calculate_boman_index(sequence: str) -> float:
    """
    Computes Boman index (kcal/mol), a measure of protein-binding potential.

    Boman index = (Σ ΔG_i) / N

    where ΔG_i is the Boman free energy scale for residue i.
    Index > 2.48 kcal/mol → high protein interaction potential.

    Reference: Boman, J. Intern. Med. 254:197-215 (2003).
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    total = sum(BOMAN_SCALE.get(aa, 0.0) for aa in seq)
    return round(total / len(seq), 2)


def calculate_conformational_entropy_penalty(
    sequence: str,
    conformation: str = "helix",
    is_cyclic: bool = False
) -> float:
    """
    Estimates the conformational entropy loss (-TΔS_conf in kcal/mol) of peptide binding
    at T = 298.15 K using residue-specific backbone, sidechain rotamer, and bimolecular association statistics.

    References:
      - D'Aquino et al. (1996). The magnitude of the backbone conformational entropy change. Proteins, 25(2), 267-278.
      - Pickett, S. D., & Sternberg, M. J. (1993). Empirical calculation of the loss of side-chain conformational entropy on protein folding and binding. J. Mol. Biol., 231(3), 825-839.
      - Creamer, T. P. (2000). Backbone conformational entropy in protein folding. Biochemistry, 39, 14214-14221.
      - Gilson, M. K. et al. (1997). The statistical-thermodynamic basis for computation of binding affinities. Biophys. J., 72, 1047-1069.
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0

    # Sidechain rotatable entropy penalty per residue (kcal/mol at 298.15 K)
    SIDECHAIN_ENTROPY = {
        'G': 0.00, 'A': 0.00, 'P': 0.00,
        'S': 0.15, 'T': 0.15, 'C': 0.15, 'V': 0.18,
        'D': 0.25, 'N': 0.25, 'L': 0.25, 'I': 0.25, 'F': 0.25, 'Y': 0.25, 'H': 0.25, 'W': 0.25,
        'E': 0.35, 'Q': 0.35, 'M': 0.35,
        'K': 0.45, 'R': 0.45
    }

    # Backbone entropy loss: ~0.22 kcal/mol per flexible residue; Glycine ~0.45 kcal/mol; Proline ~0.05 kcal/mol
    delta_s_bb = sum(0.05 if aa == 'P' else (0.45 if aa == 'G' else 0.22) for aa in seq)
    delta_s_sc = sum(SIDECHAIN_ENTROPY.get(aa, 0.20) * 0.85 for aa in seq)

    total_entropy_penalty = delta_s_bb + delta_s_sc

    # Structural pre-organization reductions:
    conf_lower = conformation.lower()
    if is_cyclic:
        total_entropy_penalty *= 0.50
    elif "ppii" in conf_lower:
        total_entropy_penalty *= 0.75   # Polyproline II helix
    elif "helix" in conf_lower:
        total_entropy_penalty *= 0.65   # Alpha-helix
    elif "sheet" in conf_lower or "extended" in conf_lower:
        total_entropy_penalty *= 0.60   # Extended beta-strand

    # Bimolecular standard-state association (cratic) entropy penalty (~0.80 kcal/mol, Gilson 1997)
    total_entropy_penalty += 0.80

    return round(float(total_entropy_penalty), 2)


def predict_hemolytic_risk_and_selectivity(
    net_charge: float,
    mean_hydrophobicity: float,
    hydrophobic_moment: float
) -> Dict[str, str]:
    """
    Evaluates antimicrobial therapeutic selectivity and human hemolytic risk.

    Therapeutic Window Criteria:
      - Bacterial Selectivity: favored by net cationic charge (z >= +2.0) targeting
        anionic bacterial membranes (phosphatidylglycerol / lipid A / LPS).
      - Hemolytic Toxicity: mammalian erythrocyte membranes are zwitterionic;
        extreme hydrophobicity (> 0.40) combined with strong amphipathicity (muH > 0.50)
        causes non-specific eukaryotic cell lysis (e.g. Melittin-like toxicity).

    References:
      - Fjell, C. D. et al. (2012). Designing antimicrobial peptides: form follows function.
        Nature Reviews Drug Discovery, 11(1), 37-51.
      - Wang, G. et al. (2016). APD3: the antimicrobial peptide database. Nucleic Acids Res.
    """
    # 1. Hemolytic Risk Evaluation
    if mean_hydrophobicity > 0.42 and hydrophobic_moment > 0.50:
        hemolytic_risk = "High (Hemolytic)"
    elif mean_hydrophobicity > 0.30 or hydrophobic_moment > 0.45:
        hemolytic_risk = "Moderate"
    else:
        hemolytic_risk = "Low (Non-hemolytic)"

    # 2. Therapeutic Selectivity Index (TSI)
    if net_charge >= 2.0 and hemolytic_risk == "Low (Non-hemolytic)":
        selectivity = "Optimal (High Selectivity)"
    elif net_charge >= 1.0 and hemolytic_risk != "High (Hemolytic)":
        selectivity = "Moderate"
    else:
        selectivity = "Low / Non-specific"

    return {
        "hemolytic_risk": hemolytic_risk,
        "therapeutic_selectivity": selectivity
    }


def estimate_solution_affinity(
    affinity_kcal_mol: float,
    sequence: str,
    conformation: str = "helix",
    is_cyclic: bool = False,
    temp_kelvin: float = 298.15
) -> Dict[str, Any]:
    """
    Computes solution-phase binding free energy (Delta G_solution) and estimated in vitro Kd
    by applying the conformational entropy loss correction:
      Delta G_solution = Delta G_dock + Delta G_entropy_loss
    Bridges the physical gap between rigid crystal docking and wet-lab buffer experiments.
    """
    penalty = calculate_conformational_entropy_penalty(sequence, conformation, is_cyclic)
    solution_dg = affinity_kcal_mol + penalty

    # In solution, if Delta G >= 0, binding is weak/non-specific
    r = 1.9872e-3
    if solution_dg >= 0.0:
        sol_kd_str = "> 1.0 mM"
    else:
        kd_molar = math.exp(solution_dg / (r * temp_kelvin))
        if kd_molar < 1e-9:
            sol_kd_str = f"{kd_molar * 1e12:.1f} pM"
        elif kd_molar < 1e-6:
            sol_kd_str = f"{kd_molar * 1e9:.1f} nM"
        elif kd_molar < 1e-3:
            sol_kd_str = f"{kd_molar * 1e6:.1f} µM"
        else:
            sol_kd_str = f"{kd_molar * 1e3:.1f} mM"

    return {
        "entropy_penalty": penalty,
        "solution_delta_g": round(float(solution_dg), 2),
        "solution_kd": sol_kd_str
    }


def _compute_pairwise_contacts(
    rec_atoms: List[Dict[str, Any]],
    lig_atoms: List[Dict[str, Any]],
    salt_bridge_dist: float = 4.0,
    hbond_dist: float = 3.5,
    hydrophobic_dist: float = 4.5,
    clash_dist: float = 1.9
) -> Dict[str, Any]:
    """Helper to compute physical contacts between two strictly disjoint atom sets."""
    if not rec_atoms or not lig_atoms:
        return {"salt_bridges": 0, "hbonds": 0, "hydrophobic_contacts": 0,
                "clashes": 0, "key_contacts": []}

    num_salt_bridges = 0
    num_hbonds = 0
    num_hydrophobic = 0
    num_clashes = 0
    contact_pairs = []

    _pos_sb_atoms = {
        "ARG": {"NE", "NH1", "NH2"},
        "LYS": {"NZ"},
        "HIP": {"ND1", "NE2"}
    }
    _neg_sb_atoms = {
        "ASP": {"OD1", "OD2"},
        "GLU": {"OE1", "OE2"}
    }
    _polar_elements = {"N", "O"}

    for latom in lig_atoms:
        lpos = latom["pos"]
        lelem = latom["element"]
        lres = latom["res_name"]
        latom_name = latom["atom_name"]

        for ratom in rec_atoms:
            rpos = ratom["pos"]
            relem = ratom["element"]
            rres = ratom["res_name"]
            ratom_name = ratom["atom_name"]

            dist = float(np.linalg.norm(lpos - rpos))

            # ── 1. Clash check (<= 1.9 A for non-H heavy atom pairs) ───────
            if dist <= clash_dist:
                num_clashes += 1
                continue

            # ── 2. Salt Bridge (Barlow & Thornton standard: 2.4 A <= d <= 4.0 A) ──
            if 2.4 <= dist <= salt_bridge_dist:
                lig_is_pos = lres in _pos_sb_atoms and latom_name in _pos_sb_atoms.get(lres, set())
                lig_is_neg = lres in _neg_sb_atoms and latom_name in _neg_sb_atoms.get(lres, set())
                rec_is_pos = rres in _pos_sb_atoms and ratom_name in _pos_sb_atoms.get(rres, set())
                rec_is_neg = rres in _neg_sb_atoms and ratom_name in _neg_sb_atoms.get(rres, set())

                is_salt = (lig_is_pos and rec_is_neg) or (lig_is_neg and rec_is_pos)
                if is_salt:
                    num_salt_bridges += 1
                    contact_pairs.append(
                        f"{lres}:{latom_name}--{ratom['chain']}:{rres}{ratom['res_num']}"
                    )

            # ── 3. Hydrogen Bond (donor-acceptor heavy atoms: 2.4 A <= d <= 3.5 A) ──
            if 2.4 <= dist <= hbond_dist:
                if lelem in _polar_elements and relem in _polar_elements:
                    l_is_carbonyl_o = (latom_name == "O" and lelem == "O")
                    r_is_carbonyl_o = (ratom_name == "O" and relem == "O")
                    if not (l_is_carbonyl_o and r_is_carbonyl_o):
                        num_hbonds += 1

            # ── 4. Hydrophobic Contact (nonpolar C-C pairs: 2.8 A <= d <= 4.5 A) ──
            if 2.8 <= dist <= hydrophobic_dist:
                lig_hydrophobic = lres in _HYDROPHOBIC_RESIDUES and lelem == "C"
                rec_hydrophobic = rres in _HYDROPHOBIC_RESIDUES and relem == "C"
                if lig_hydrophobic and rec_hydrophobic:
                    num_hydrophobic += 1

    return {
        "salt_bridges": num_salt_bridges,
        "hbonds": num_hbonds,
        "hydrophobic_contacts": num_hydrophobic,
        "clashes": num_clashes,
        "key_contacts": list(set(contact_pairs))[:10]
    }


def analyze_cocomplex_contacts(
    complex_pdb: Path,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
    salt_bridge_dist: float = 4.0,
    hbond_dist: float = 3.5,
    hydrophobic_dist: float = 4.5,
    clash_dist: float = 1.9
) -> Dict[str, Any]:
    """
    Analyzes strictly cross-chain inter-molecular contacts within a 2-chain co-complex PDB.
    Guarantees that receptor_chain != peptide_chain and eliminates any intra-chain
    or peptide<->peptide self-contacts.
    """
    if receptor_chain == peptide_chain:
        logger.warning(
            f"analyze_cocomplex_contacts called with identical receptor and peptide chains ('{receptor_chain}'). "
            "Self-contacts are strictly excluded."
        )
        return {"salt_bridges": 0, "hbonds": 0, "hydrophobic_contacts": 0, "clashes": 0, "key_contacts": []}

    complex_pdb = Path(complex_pdb)
    if not complex_pdb.exists():
        logger.debug(f"Complex PDB not found: {complex_pdb}")
        return {"salt_bridges": 0, "hbonds": 0, "hydrophobic_contacts": 0, "clashes": 0, "key_contacts": []}

    try:
        st = gemmi.read_structure(str(complex_pdb))
    except Exception as e:
        logger.debug(f"Could not read structure {complex_pdb}: {e}")
        return {"salt_bridges": 0, "hbonds": 0, "hydrophobic_contacts": 0, "clashes": 0, "key_contacts": []}

    rec_atoms = []
    pep_atoms = []

    for model in st:
        for chain in model:
            if chain.name == receptor_chain:
                for res in chain:
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        rec_atoms.append({
                            "res_name": res.name,
                            "res_num": res.seqid.num,
                            "chain": chain.name,
                            "atom_name": atom.name,
                            "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                            "element": atom.element.name
                        })
            elif chain.name == peptide_chain:
                for res in chain:
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        pep_atoms.append({
                            "res_name": res.name,
                            "res_num": res.seqid.num,
                            "chain": chain.name,
                            "atom_name": atom.name,
                            "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                            "element": atom.element.name
                        })

    return _compute_pairwise_contacts(
        rec_atoms, pep_atoms,
        salt_bridge_dist=salt_bridge_dist,
        hbond_dist=hbond_dist,
        hydrophobic_dist=hydrophobic_dist,
        clash_dist=clash_dist
    )


def analyze_complex_contacts(
    receptor_pdb: Path,
    docked_pose_pdb: Path,
    salt_bridge_dist: float = 4.0,
    hbond_dist: float = 3.5,
    hydrophobic_dist: float = 4.5,
    clash_dist: float = 1.9,
    receptor_chain: str = "A",
    peptide_chain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes inter-molecular contacts between target receptor and docked peptide pose.
    Supports either pure peptide PDBs (e.g. from Vina) or 2-chain co-complex PDBs.
    """
    # If the exact same file is passed as both receptor and ligand, delegate to analyze_cocomplex_contacts
    try:
        if Path(receptor_pdb).resolve() == Path(docked_pose_pdb).resolve():
            target_pep = peptide_chain or ("B" if receptor_chain == "A" else "A")
            return analyze_cocomplex_contacts(
                receptor_pdb,
                receptor_chain=receptor_chain,
                peptide_chain=target_pep,
                salt_bridge_dist=salt_bridge_dist,
                hbond_dist=hbond_dist,
                hydrophobic_dist=hydrophobic_dist,
                clash_dist=clash_dist
            )
    except Exception:
        pass

    try:
        rec_doc = gemmi.read_structure(str(receptor_pdb))
        lig_doc = gemmi.read_structure(str(docked_pose_pdb))
    except Exception as e:
        logger.debug(f"Could not read structures for contact analysis: {e}")
        return {
            "salt_bridges": 0,
            "hbonds": 0,
            "hydrophobic_contacts": 0,
            "clashes": 0,
            "key_contacts": []
        }

    # Extract receptor heavy atoms (prefer specified receptor chain if present)
    rec_atoms = []
    has_target_rec_chain = any(c.name == receptor_chain for m in rec_doc for c in m)
    for model in rec_doc:
        for chain in model:
            if has_target_rec_chain and chain.name != receptor_chain:
                continue
            for res in chain:
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    rec_atoms.append({
                        "res_name": res.name,
                        "res_num": res.seqid.num,
                        "chain": chain.name,
                        "atom_name": atom.name,
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                        "element": atom.element.name
                    })

    # Extract peptide heavy atoms (model 1)
    lig_atoms = []
    if len(lig_doc) > 0:
        model = lig_doc[0]
        chain_names = [c.name for c in model]

        # Determine target peptide chain
        target_pep_chain = peptide_chain
        if not target_pep_chain:
            if len(chain_names) > 1:
                # If complex with multiple chains, select chain other than receptor
                other_chains = [c for c in chain_names if c != receptor_chain]
                target_pep_chain = other_chains[0] if other_chains else None

        for chain in model:
            if target_pep_chain and chain.name != target_pep_chain:
                continue
            for res in chain:
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    lig_atoms.append({
                        "res_name": res.name,
                        "res_num": res.seqid.num,
                        "chain": chain.name,
                        "atom_name": atom.name,
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                        "element": atom.element.name
                    })

    return _compute_pairwise_contacts(
        rec_atoms, lig_atoms,
        salt_bridge_dist=salt_bridge_dist,
        hbond_dist=hbond_dist,
        hydrophobic_dist=hydrophobic_dist,
        clash_dist=clash_dist
    )


def compute_composite_score(
    affinity_kcal_mol: float,
    net_charge: float,
    hydrophobic_moment: float,
    boman_index: float,
    salt_bridges: int,
    hbonds: int,
    clashes: int
) -> float:
    """
    Composite AMP-Target Interaction Score (ATIS).

    Balances binding affinity (favorable negative ΔG) with AMP biophysical fitness
    and intermolecular anchor contacts. All contributions are additive with weights
    calibrated to AMP literature (Wang et al. APD3; Fjell et al. Nat. Rev. Drug Discov. 2012).

    Score composition:
    - Vina ΔG (negated, so high affinity = high score)
    - Salt bridge bonus: 1.5 per bridge (anchors charged contacts)
    - H-bond bonus: 0.5 per bond (up to 8 bonds; returns diminish quickly)
    - Amphipathic moment bonus: 2.0 × μH (rewards membrane-active AMPs)
    - Boman index bonus: 0.4 × BI (rewards protein-interaction potential)
    - Clash penalty: 3.0 per clash (severe penalty for steric violations)
    - Charge penalty: -2.0 if net charge < +1.0 (most active AMPs are cationic)
    """
    score = -1.0 * affinity_kcal_mol          # Higher affinity (more negative ΔG) → higher score
    score += 1.5 * salt_bridges               # Electrostatic anchor bonus
    score += 0.5 * min(hbonds, 8)             # H-bond bonus (capped at 8)
    score += 2.0 * hydrophobic_moment          # Amphipathic bonus
    score += 0.4 * max(0.0, boman_index)      # Protein-interaction potential bonus
    score -= 3.0 * clashes                     # Steric clash penalty (strong)
    if net_charge < 1.0:
        score -= 2.0                           # Charge state penalty

    return round(score, 2)


def count_heavy_atoms(sequence: str) -> int:
    """
    Counts estimated heavy atoms (C, N, O, S) per standard peptide sequence.

    Counts are per standard amino acid residue (backbone + sidechain heavy atoms),
    excluding H. The C-terminal oxygen is included (+1).

    Reference: RCSB PDB atom count tables for standard residues.
    """
    HEAVY_ATOMS = {
        'A':  5, 'R': 11, 'N':  8, 'D':  8, 'C':  6,
        'Q':  9, 'E':  9, 'G':  4, 'H': 10, 'I':  8,
        'L':  8, 'K':  9, 'M':  8, 'F': 11, 'P':  7,
        'S':  6, 'T':  7, 'W': 14, 'Y': 12, 'V':  7
    }
    seq = clean_sequence(sequence)
    # +1 for the C-terminal hydroxyl oxygen (not included in sidechain counts above)
    return sum(HEAVY_ATOMS.get(aa, 7) for aa in seq) + 1


def calculate_ligand_efficiency(affinity_kcal_mol: float, num_heavy_atoms: int) -> float:
    """
    Ligand Efficiency (LE) = -ΔG / N_heavy  (kcal/mol per heavy atom).

    A value of 0.3 kcal/mol/atom is considered efficient for drug-like molecules.
    For peptides, LE is typically 0.15–0.25 (lower due to more heavy atoms).

    Reference: Hopkins et al., Drug Discov. Today 9:430-431 (2004).
    """
    if num_heavy_atoms <= 0:
        return 0.0
    return round(-1.0 * affinity_kcal_mol / num_heavy_atoms, 3)


def calculate_per_residue_energy(affinity_kcal_mol: float, num_residues: int) -> float:
    """
    Per-Residue Binding Energy (BEPR) = ΔG / N_res  (kcal/mol per residue).

    More negative value → tighter binding per residue.
    Typical range for peptides: -1.0 to -1.5 kcal/mol/residue.
    """
    if num_residues <= 0:
        return 0.0
    return round(affinity_kcal_mol / num_residues, 3)
