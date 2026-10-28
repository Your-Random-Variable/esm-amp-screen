"""
Machine Learning Scoring Engine for Protein-Peptide Interactions.
Combines:
1. RF-Score-Peptide atomic interaction fingerprints (20 element-pair distance bins, normalized by peptide size)
2. HADDOCK-style explicit physics terms (vdW, electrostatics, desolvation, BSA, HADDOCK score, AIR restraints)
3. AMP biophysical properties (net charge, amphipathic moment, Boman index)
4. Calibrated Ensemble Regressors (Random Forest + Gradient Boosting) for pKd and Delta G prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import cdist
import gemmi
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

from esm_amp_screen.utils import logger, AA_3TO1
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import (
    analyze_cocomplex_contacts,
    calculate_net_charge,
    calculate_hydrophobic_moment,
    calculate_boman_index
)

# Complete set of 4 biological elements -> 10 unique unordered pairs
ELEMENTS = ["C", "N", "O", "S"]
ELEMENT_PAIRS = [
    ("C", "C"), ("C", "N"), ("C", "O"), ("C", "S"),
    ("N", "N"), ("N", "O"), ("N", "S"),
    ("O", "O"), ("O", "S"), ("S", "S")
]

# Physical constants for pKd <-> Delta G conversion (Gibbs-Helmholtz / van 't Hoff)
R_GAS_KCAL = 0.001987204   # Gas constant R in kcal / (mol * K)
TEMP_KELVIN = 298.15       # Standard physiological temperature (25 °C)
RT_LN10 = R_GAS_KCAL * TEMP_KELVIN * np.log(10.0)  # ~ 1.3638 kcal/mol per pKd unit


def extract_structural_fingerprint(
    complex_pdb: Path,
    receptor_chain: str = "A",
    peptide_chain: str = "B"
) -> Dict[str, float]:
    """
    Extracts 20 atomic contact features across all 10 standard element-pair combinations:
    - 10 close contacts (d <= 4.0 A) normalized by peptide heavy atom count
    - 10 medium-range shell contacts (4.0 A < d <= 8.0 A) normalized by peptide heavy atom count
    Normalization by peptide heavy atoms removes spurious length bias.
    """
    st = gemmi.read_structure(str(complex_pdb))

    rec_coords = []
    rec_elems = []
    pep_coords = []
    pep_elems = []

    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    el = atom.element.name
                    if el not in ELEMENTS:
                        continue
                    pt = [atom.pos.x, atom.pos.y, atom.pos.z]
                    if chain.name == receptor_chain:
                        rec_coords.append(pt)
                        rec_elems.append(el)
                    elif chain.name == peptide_chain:
                        pep_coords.append(pt)
                        pep_elems.append(el)

    features = {}
    for e1, e2 in ELEMENT_PAIRS:
        features[f"pair_{e1}_{e2}_close"] = 0.0
        features[f"pair_{e1}_{e2}_medium"] = 0.0

    if not rec_coords or not pep_coords:
        return features

    dists = cdist(np.array(rec_coords), np.array(pep_coords))
    norm_factor = max(1.0, float(len(pep_coords)))

    for i, r_el in enumerate(rec_elems):
        close_j = np.where(dists[i] <= 4.0)[0]
        med_j = np.where((dists[i] > 4.0) & (dists[i] <= 8.0))[0]

        for j in close_j:
            p_el = pep_elems[j]
            pair_key = (r_el, p_el) if (r_el, p_el) in ELEMENT_PAIRS else (p_el, r_el)
            if pair_key in ELEMENT_PAIRS:
                features[f"pair_{pair_key[0]}_{pair_key[1]}_close"] += 1.0 / norm_factor

        for j in med_j:
            p_el = pep_elems[j]
            pair_key = (r_el, p_el) if (r_el, p_el) in ELEMENT_PAIRS else (p_el, r_el)
            if pair_key in ELEMENT_PAIRS:
                features[f"pair_{pair_key[0]}_{pair_key[1]}_medium"] += 1.0 / norm_factor

    return features


class MLPeptideScorer:
    """
    Multi-Parametric Consensus & Machine Learning Scorer for Protein-Peptide Affinity.
    Integrates RF-Score structural fingerprints with HADDOCK physics & AMP descriptors.
    When a pre-trained model file (.pkl / .joblib) is provided, uses the trained regressor.
    When no pre-trained model is supplied, transparently marks is_calibrated = False and
    computes an uncalibrated heuristic consensus score based on explicit physics and docking terms.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.rf_model = None
        self.gb_model = None
        self.feature_names = []
        self.is_calibrated = False

        if model_path and Path(model_path).exists():
            self._load_model(Path(model_path))
            self.is_calibrated = True
        else:
            self._setup_features()
            self.is_calibrated = False

    def _load_model(self, model_path: Path):
        """Loads a pre-trained Random Forest and Gradient Boosting model from disk."""
        data = joblib.load(str(model_path))
        if isinstance(data, dict):
            self.rf_model = data.get("rf")
            self.gb_model = data.get("gb")
            self.feature_names = data.get("features", [])
            logger.info(
                f"Loaded pre-trained ML model from {model_path.name} "
                f"(RF + GB, {len(self.feature_names)} features)"
            )
        else:
            raise ValueError(
                f"Invalid model file format in {model_path}. Expected dictionary with 'rf' and 'gb' keys."
            )

    def save_model(self, out_path: Path):
        """Saves current trained ensemble models to disk with joblib."""
        out_path = Path(out_path)
        joblib.dump(
            {"rf": self.rf_model, "gb": self.gb_model, "features": self.feature_names},
            str(out_path)
        )
        logger.info(f"Saved trained ML model to {out_path.name}")

    def _setup_features(self):
        """Sets up feature names for 34D structural and physical interface fingerprints."""
        feature_names = []
        for e1, e2 in ELEMENT_PAIRS:
            feature_names.append(f"pair_{e1}_{e2}_close")
            feature_names.append(f"pair_{e1}_{e2}_medium")
        feature_names.extend([
            "e_vdw", "e_elec", "e_desolv", "bsa", "haddock_score", "e_air",
            "salt_bridges", "hbonds", "hydrophobic_contacts", "clashes",
            "net_charge", "hydrophobic_moment", "boman_index", "length"
        ])
        self.feature_names = feature_names

    def compute_consensus_score(
        self,
        docking_energy: Optional[float] = None,
        haddock_score: float = -40.0,
        sol_delta_g: Optional[float] = None,
        **kwargs
    ) -> float:
        """Balanced consensus scoring combining docking, explicit physics, and solution thermodynamics."""
        dock_val = docking_energy if docking_energy is not None else kwargs.get("vina_affinity", -8.0)
        dock_norm = -float(dock_val)
        haddock_norm = -float(haddock_score) / 10.0
        sol_dg = sol_delta_g if sol_delta_g is not None else float(dock_val)
        ml_pkd = max(3.0, min(12.0, -sol_dg / RT_LN10))
        return float((0.35 * dock_norm) + (0.35 * haddock_norm) + (0.30 * ml_pkd))

    def score_complex(
        self,
        complex_pdb: Path,
        hit_meta: Optional[Dict[str, Any]] = None,
        pocket_residues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculates HADDOCK physics, RF structural fingerprints, and ML-predicted affinity.
        Returns full dict of physical & machine-learned metrics without label leakage.
        Automatically extracts interface contacts and sequence properties if missing from hit_meta.
        """
        complex_pdb = Path(complex_pdb)
        if hit_meta is None:
            hit_meta = {}
        else:
            hit_meta = dict(hit_meta)

        # 1. Physics Engine Evaluation
        physics = calculate_haddock_physics_score(
            complex_pdb=complex_pdb,
            pocket_residues=pocket_residues
        )

        # 2. Structural Fingerprint
        fp = extract_structural_fingerprint(complex_pdb)

        # 3. Intermolecular contacts (auto-extract if missing)
        if any(k not in hit_meta for k in ("salt_bridges", "hbonds", "hydrophobic_contacts", "clashes")):
            try:
                c_data = analyze_cocomplex_contacts(complex_pdb, receptor_chain="A", peptide_chain="B")
                for k, v in c_data.items():
                    if k not in hit_meta:
                        hit_meta[k] = v
            except Exception:
                pass

        # 4. Sequence & AMP biophysics (auto-extract if missing)
        seq = hit_meta.get("sequence")
        if not seq:
            try:
                st = gemmi.read_structure(str(complex_pdb))
                seq = "".join(
                    AA_3TO1.get(res.name.upper(), "")
                    for m in st for c in m if c.name == "B"
                    for res in c if res.name.upper() in AA_3TO1
                )
                if seq:
                    hit_meta["sequence"] = seq
            except Exception:
                pass

        if seq:
            if "net_charge_z" not in hit_meta and "net_charge" not in hit_meta:
                hit_meta["net_charge_z"] = calculate_net_charge(seq)
            if "hydrophobic_moment_muH" not in hit_meta and "hydrophobic_moment" not in hit_meta:
                hit_meta["hydrophobic_moment_muH"] = calculate_hydrophobic_moment(seq)
            if "boman_index" not in hit_meta:
                hit_meta["boman_index"] = calculate_boman_index(seq)

        pep_len = float(hit_meta.get("length", len(seq) if seq else 12.0))
        dock_energy = float(hit_meta.get("affinity", hit_meta.get("docking_energy", physics["haddock_score"] * 0.28)))

        # Assemble Feature Vector (exactly 34 dimensions, pure physical chemistry)
        feat_vec = []
        for e1, e2 in ELEMENT_PAIRS:
            feat_vec.append(fp.get(f"pair_{e1}_{e2}_close", 0.0))
            feat_vec.append(fp.get(f"pair_{e1}_{e2}_medium", 0.0))

        feat_vec.extend([
            physics["e_vdw"],
            physics["e_elec"],
            physics["e_desolv"],
            physics["bsa"],
            physics["haddock_score"],
            physics["e_air"],
            float(hit_meta.get("salt_bridges", 0)),
            float(hit_meta.get("hbonds", 0)),
            float(hit_meta.get("hydrophobic_contacts", 0)),
            float(hit_meta.get("clashes", 0)),
            float(hit_meta.get("net_charge_z", hit_meta.get("net_charge", 0.0))),
            float(hit_meta.get("hydrophobic_moment_muH", hit_meta.get("hydrophobic_moment", 0.0))),
            float(hit_meta.get("boman_index", 0.0)),
            pep_len
        ])

        X_input = np.array([feat_vec])

        # 4. Ensemble ML / Consensus Predictions
        if self.is_calibrated and self.rf_model is not None and self.gb_model is not None:
            pred_rf = float(self.rf_model.predict(X_input)[0])
            pred_gb = float(self.gb_model.predict(X_input)[0])
            ml_pkd = (pred_rf + pred_gb) / 2.0
            variance_penalty = abs(pred_rf - pred_gb) * 10.0
            clash_penalty = hit_meta.get("clashes", 0) * 15.0
            bsa_reward = min(30.0, (physics["bsa"] / 1000.0) * 30.0)
            confidence = float(np.clip(70.0 + bsa_reward - variance_penalty - clash_penalty, 10.0, 95.0))
        else:
            # Transparent physical consensus derived from solution thermodynamics & docking
            sol_dg = float(hit_meta.get("solution_delta_g", dock_energy))
            ml_pkd = max(3.0, min(12.0, -sol_dg / RT_LN10))
            clash_pen = min(40.0, hit_meta.get("clashes", 0) * 10.0)
            bsa_score = min(30.0, (physics["bsa"] / 1000.0) * 30.0)
            contacts_score = min(20.0, (hit_meta.get("hbonds", 0) + hit_meta.get("salt_bridges", 0) * 2) * 3.0)
            confidence = float(np.clip(50.0 + bsa_score + contacts_score - clash_pen, 20.0, 85.0))

        # Exact Gibbs free energy conversion: Delta G = -RT * ln(10) * pKd (kcal/mol at 298.15 K)
        ml_delta_g = -RT_LN10 * ml_pkd

        # Calibrated Affinity Tiers (replaces indefensible sub-picomolar pseudo-precision)
        if ml_pkd >= 8.0:
            affinity_tier = "Tier 1: High Affinity (< 10 nM, Top 1%)"
        elif ml_pkd >= 6.5:
            affinity_tier = "Tier 2: Moderate Affinity (10-300 nM, Top 5%)"
        elif ml_pkd >= 5.0:
            affinity_tier = "Tier 3: Micromolar Binding (0.3-10 µM, Top 20%)"
        else:
            affinity_tier = "Tier 4: Weak / Non-Binder (> 10 µM)"

        kd_molar = 10.0 ** (-ml_pkd)
        if kd_molar < 1e-8:
            kd_str = "< 10 nM (Strong)"
        elif kd_molar < 1e-6:
            kd_str = f"{kd_molar * 1e9:.0f} nM"
        elif kd_molar < 1e-3:
            kd_str = f"{kd_molar * 1e6:.1f} µM"
        else:
            kd_str = f"{kd_molar * 1e3:.1f} mM"

        # Unified Balanced Consensus Score (combining geometric docking, HADDOCK, and solution thermodynamics)
        dock_norm = -float(dock_energy)
        haddock_norm = -float(physics["haddock_score"]) / 10.0
        ml_norm = float(ml_pkd)
        consensus_score = (0.35 * dock_norm) + (0.35 * haddock_norm) + (0.30 * ml_norm)

        return {
            "haddock_score": physics["haddock_score"],
            "e_vdw": physics["e_vdw"],
            "e_elec": physics["e_elec"],
            "e_desolv": physics["e_desolv"],
            "bsa": physics["bsa"],
            "e_air": physics["e_air"],
            "ml_pkd": round(float(ml_pkd), 2),
            "ml_delta_g": round(float(ml_delta_g), 2),
            "ml_kd": kd_str,
            "affinity_tier": affinity_tier,
            "confidence_score": round(float(confidence), 1),
            "consensus_score": round(float(consensus_score), 2),
            "is_calibrated": self.is_calibrated
        }
