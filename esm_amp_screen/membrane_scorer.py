"""
Membrane Biophysics and Antimicrobial Peptide (AMP) Scoring Engine.

Models the primary biological mechanism of antimicrobial peptides:
interfacial binding, lipid bilayer partitioning, amphipathic moment,
and membrane permeabilization using experimentally measured thermodynamic scales:
1. Wimley-White Interfacial Hydrophobicity Scale (ΔG_w->if, kcal/mol)
   [Wimley & White, Nature Struct. Biol. 3:842-848 (1996); Biochemistry 35:5109 (1996)]
2. Wimley-White Octanol Transfer Free Energy Scale (ΔG_w->oct, kcal/mol)
3. Eisenberg Amphipathic Hydrophobic Moment (μH) for α-helices (100°) and β-sheets (160°/180°)
4. Cationic Surface Charge Density (Z / Length at physiological pH 7.4)
5. Membrane Insertion Index and In Silico Therapeutic Index (Selectivity over Hemolysis)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Wimley-White Whole-Residue Interfacial Free Energy Scale (ΔG_w->if in kcal/mol at 25 °C)
# Negative values favor partitioning from aqueous phase into the phosphocholine bilayer interface.
# Trp and Phe are the primary interfacial membrane anchors.
WIMLEY_WHITE_INTERFACIAL: Dict[str, float] = {
    "W": -1.85,  # Strongest aromatic anchor at lipid-water interface
    "F": -1.13,
    "Y": -0.94,
    "L": -0.56,
    "I": -0.31,
    "C": -0.24,
    "M": -0.23,
    "V": 0.07,
    "G": 0.01,
    "S": 0.13,
    "T": 0.14,
    "A": 0.17,
    "P": 0.45,
    "N": 0.42,
    "Q": 0.58,
    "H": 0.17,   # Neutral state at physiological pH
    "R": 0.81,   # Charged guanidinium partitioned into polar headgroups
    "K": 0.99,   # Charged ammonium partitioned into polar headgroups
    "D": 1.23,   # Ionized carboxylate (deprotonated at pH 7.4)
    "E": 2.02    # Ionized carboxylate (deprotonated at pH 7.4)
}

# Wimley-White Octanol Transfer Scale (ΔG_w->oct in kcal/mol)
# Represents partitioning into the non-polar hydrocarbon core of the membrane.
WIMLEY_WHITE_OCTANOL: Dict[str, float] = {
    "W": -2.09,
    "F": -1.71,
    "L": -1.25,
    "I": -1.12,
    "M": -0.67,
    "V": -0.46,
    "Y": -0.71,
    "C": -0.02,
    "A": 0.50,
    "T": 0.25,
    "S": 0.46,
    "G": 1.15,
    "P": 0.14,
    "H": 0.11,
    "Q": 0.77,
    "N": 0.85,
    "E": 3.63,
    "D": 3.64,
    "K": 2.80,
    "R": 1.81
}

# Eisenberg Consensus Hydrophobicity Scale (normalized for amphipathic moment)
EISENBERG_CONSENSUS: Dict[str, float] = {
    "I": 0.73, "F": 0.61, "V": 0.54, "L": 0.53, "W": 0.37,
    "M": 0.26, "A": 0.25, "G": 0.16, "C": 0.04, "Y": 0.02,
    "P": -0.07, "T": -0.18, "S": -0.26, "H": -0.40, "E": -0.62,
    "N": -0.64, "Q": -0.69, "D": -0.72, "K": -1.10, "R": -1.76
}

# pKa values for Henderson-Hasselbalch net charge computation
PKA_VALUES = {
    "N_term": 9.69,
    "C_term": 2.34,
    "C": 8.33,
    "D": 3.86,
    "E": 4.25,
    "H": 6.00,
    "K": 10.53,
    "R": 12.48,
    "Y": 10.07
}


def clean_amp_sequence(seq: str) -> str:
    """Standardizes amino acid sequence, uppercase, stripping whitespace."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join([aa for aa in seq.upper() if aa in valid_aa])
    return cleaned


def calculate_net_charge_at_ph(seq: str, ph: float = 7.4) -> float:
    """
    Computes exact net charge using the Henderson-Hasselbalch equation at specified pH.
    """
    seq = clean_amp_sequence(seq)
    if not seq:
        return 0.0

    # N-terminus
    charge = 1.0 / (1.0 + 10.0 ** (ph - PKA_VALUES["N_term"]))
    # C-terminus
    charge -= 1.0 / (1.0 + 10.0 ** (PKA_VALUES["C_term"] - ph))

    for aa in seq:
        if aa in ("K", "R"):
            pka = PKA_VALUES[aa]
            charge += 1.0 / (1.0 + 10.0 ** (ph - pka))
        elif aa == "H":
            pka = PKA_VALUES["H"]
            charge += 1.0 / (1.0 + 10.0 ** (ph - pka))
        elif aa in ("D", "E", "C", "Y"):
            pka = PKA_VALUES[aa]
            charge -= 1.0 / (1.0 + 10.0 ** (pka - ph))

    return float(charge)


def calculate_wimley_white_interfacial_dg(seq: str) -> Dict[str, float]:
    """
    Computes total and per-residue Wimley-White interfacial transfer free energy (ΔG_w->if).
    Negative values favor spontaneous accumulation at lipid membrane interfaces.
    """
    seq = clean_amp_sequence(seq)
    if not seq:
        return {"total_dg_if": 0.0, "mean_dg_if": 0.0, "aromatic_anchor_count": 0}

    total_dg = sum(WIMLEY_WHITE_INTERFACIAL.get(aa, 0.0) for aa in seq)
    aromatic_anchors = sum(1 for aa in seq if aa in ("W", "F", "Y"))

    return {
        "total_dg_if": round(float(total_dg), 2),
        "mean_dg_if": round(float(total_dg / len(seq)), 3),
        "aromatic_anchor_count": aromatic_anchors
    }


def calculate_wimley_white_octanol_dg(seq: str) -> Dict[str, float]:
    """
    Computes total and per-residue Wimley-White octanol transfer free energy (ΔG_w->oct).
    Represents insertion from water into the non-polar hydrophobic core of the bilayer.
    """
    seq = clean_amp_sequence(seq)
    if not seq:
        return {"total_dg_oct": 0.0, "mean_dg_oct": 0.0}

    total_dg = sum(WIMLEY_WHITE_OCTANOL.get(aa, 0.0) for aa in seq)
    return {
        "total_dg_oct": round(float(total_dg), 2),
        "mean_dg_oct": round(float(total_dg / len(seq)), 3)
    }


def calculate_amphipathic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """
    Calculates Eisenberg hydrophobic moment (μH) for a given periodicity angle.
    - α-helix: 100.0° (3.6 residues per turn)
    - β-sheet / hairpin: 160.0° - 180.0°
    """
    seq = clean_amp_sequence(seq)
    n = len(seq)
    if n == 0:
        return 0.0

    angle_rad = math.radians(angle_deg)
    sum_cos = 0.0
    sum_sin = 0.0

    for i, aa in enumerate(seq):
        h = EISENBERG_CONSENSUS.get(aa, 0.0)
        theta = i * angle_rad
        sum_cos += h * math.cos(theta)
        sum_sin += h * math.sin(theta)

    mu_h = math.sqrt(sum_cos ** 2 + sum_sin ** 2) / float(n)
    return round(float(mu_h), 3)


def evaluate_membrane_activity(seq: str) -> Dict[str, Any]:
    """
    Comprehensive biophysical evaluation of antimicrobial membrane permeabilization.
    Calculates:
    - Net charge Z and charge density (Z/L)
    - Wimley-White interfacial and octanol partitioning
    - Alpha-helical (100°) and Beta-sheet (160°) amphipathic moments
    - Membrane Permeabilization Index (MPI)
    - Predicted Hemolysis Risk & In Silico Therapeutic Index
    """
    seq = clean_amp_sequence(seq)
    n = len(seq)
    if n < 4:
        raise ValueError(f"Sequence too short for membrane evaluation (minimum 4 residues, got {n})")

    # 1. Charge and charge density
    z = calculate_net_charge_at_ph(seq, ph=7.4)
    charge_density = z / float(n)

    # 2. Wimley-White interfacial energetics
    ww_if = calculate_wimley_white_interfacial_dg(seq)
    ww_oct = calculate_wimley_white_octanol_dg(seq)

    # 3. Secondary structure amphipathic moments
    mu_h_alpha = calculate_amphipathic_moment(seq, angle_deg=100.0)
    mu_h_beta = calculate_amphipathic_moment(seq, angle_deg=160.0)
    best_mu_h = max(mu_h_alpha, mu_h_beta)
    favored_conf = "alpha_helix" if mu_h_alpha >= mu_h_beta else "beta_sheet"

    # 4. Membrane Permeabilization Potential (MPI)
    # Rewards: high amphipathic moment, positive charge density (selective for anionic bacterial LPS/PG),
    # and favorable interfacial partition (negative ΔG_w->if).
    # Scaled to a standard 0 to 100 potential scale.
    interfacial_score = max(0.0, -ww_if["mean_dg_if"] * 25.0)
    amphipathic_score = min(40.0, best_mu_h * 70.0)
    charge_score = min(35.0, max(0.0, charge_density * 80.0))

    mpi_score = float(np.clip(interfacial_score + amphipathic_score + charge_score, 0.0, 100.0))

    # 5. Hemolysis Risk (Safety against human erythrocytes)
    # High hydrophobicity with high octanol partition (insertion into zwitterionic mammalian membranes)
    # without cationic selectivity drives hemolysis (e.g. Melittin).
    octanol_penalty = max(0.0, -ww_oct["mean_dg_oct"] * 30.0)
    hydrophobic_fraction = sum(1 for aa in seq if aa in ("L", "I", "V", "F", "W")) / float(n)
    
    hemolytic_propensity = (octanol_penalty * 0.5) + (hydrophobic_fraction * 50.0) - (charge_density * 15.0)
    hemolytic_risk_prob = float(np.clip(1.0 / (1.0 + math.exp(-(hemolytic_propensity - 25.0) / 8.0)), 0.01, 0.99))

    if hemolytic_risk_prob < 0.25:
        hemo_tier = "Low (<10% lysis)"
    elif hemolytic_risk_prob < 0.60:
        hemo_tier = "Moderate (10-30% lysis)"
    else:
        hemo_tier = "High (>50% lysis)"

    # 6. In Silico Therapeutic Index (TI)
    p_bactericidal = float(np.clip(mpi_score / 100.0, 0.05, 0.98))
    ti = p_bactericidal / max(0.05, hemolytic_risk_prob)

    # 7. Categorization & Actionable Hit Triage
    if mpi_score >= 65.0 and hemolytic_risk_prob < 0.35:
        verdict = "Tier 1: High-Potency Selective AMP (Lead Candidate)"
    elif mpi_score >= 50.0 and hemolytic_risk_prob < 0.55:
        verdict = "Tier 2: Moderate AMP Activity (Needs Hydrophobic/Charge Tuning)"
    elif mpi_score >= 40.0:
        verdict = "Tier 3: Weak / Marginal Permeabilization"
    else:
        verdict = "Tier 4: Inactive / Poor Membrane Affinity"

    return {
        "sequence": seq,
        "length": n,
        "net_charge_z": round(z, 2),
        "charge_density_z_per_res": round(charge_density, 3),
        "total_dg_interfacial_kcal": ww_if["total_dg_if"],
        "mean_dg_interfacial_kcal": ww_if["mean_dg_if"],
        "total_dg_octanol_kcal": ww_oct["total_dg_oct"],
        "aromatic_anchors_count": ww_if["aromatic_anchor_count"],
        "mu_h_alpha_100deg": mu_h_alpha,
        "mu_h_beta_160deg": mu_h_beta,
        "favored_conformation": favored_conf,
        "membrane_permeabilization_index": round(mpi_score, 1),
        "bactericidal_probability": round(p_bactericidal, 3),
        "hemolysis_risk_probability": round(hemolytic_risk_prob, 3),
        "hemolysis_tier": hemo_tier,
        "therapeutic_index": round(float(ti), 2),
        "verdict": verdict
    }
