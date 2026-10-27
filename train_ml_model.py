"""
Model Training & Fine-Tuning Utility for ESM-AMP-Screen.
Extracts 36D structural & physical feature vectors from a dataset of
protein-peptide co-complexes with known experimental affinities (pKd),
trains the dual Random Forest + Gradient Boosting ensemble, and saves the model.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

from esm_amp_screen.ml_scorer import extract_structural_fingerprint, ELEMENT_PAIRS
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.utils import logger, ensure_dir


FEATURE_NAMES = []
for e1, e2 in ELEMENT_PAIRS:
    FEATURE_NAMES.append(f"pair_{e1}_{e2}_close")
    FEATURE_NAMES.append(f"pair_{e1}_{e2}_medium")
FEATURE_NAMES.extend([
    "e_vdw", "e_elec", "e_desolv", "bsa", "haddock_score", "e_air",
    "salt_bridges", "hbonds", "hydrophobic_contacts", "clashes",
    "net_charge", "hydrophobic_moment", "boman_index", "length",
    "vina_affinity", "delta_delta_g"
])


def featurize_complex(complex_pdb: Path, meta_row: dict) -> list:
    """Extracts 36D feature vector from a single complex PDB."""
    fp = extract_structural_fingerprint(complex_pdb)
    physics = calculate_haddock_physics_score(complex_pdb)

    row = []
    for e1, e2 in ELEMENT_PAIRS:
        row.append(fp.get(f"pair_{e1}_{e2}_close", 0.0))
        row.append(fp.get(f"pair_{e1}_{e2}_medium", 0.0))

    row.extend([
        physics["e_vdw"],
        physics["e_elec"],
        physics["e_desolv"],
        physics["bsa"],
        physics["haddock_score"],
        physics["e_air"],
        float(meta_row.get("salt_bridges", 0)),
        float(meta_row.get("hbonds", 0)),
        float(meta_row.get("hydrophobic_contacts", 0)),
        float(meta_row.get("clashes", 0)),
        float(meta_row.get("net_charge", meta_row.get("net_charge_z", 0.0))),
        float(meta_row.get("hydrophobic_moment", meta_row.get("hydrophobic_moment_muH", 0.0))),
        float(meta_row.get("boman_index", 0.0)),
        float(meta_row.get("length", 15)),
        float(meta_row.get("vina_affinity", meta_row.get("affinity", -10.0))),
        float(meta_row.get("delta_delta_g", 0.0))
    ])
    return row


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    out_model_path: Path,
    n_splits: int = 5
) -> dict:
    """Trains the ensemble and evaluates 5-fold cross-validation performance."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_cv = np.zeros_like(y)

    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)

        y_pred_cv[val_idx] = (rf.predict(X_va) + gb.predict(X_va)) / 2.0

    r_val, _ = pearsonr(y, y_pred_cv)
    rho_val, _ = spearmanr(y, y_pred_cv)
    rmse_val = float(np.sqrt(np.mean((y - y_pred_cv) ** 2)))

    # Fit final models on all data
    final_rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    final_gb = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
    final_rf.fit(X, y)
    final_gb.fit(X, y)

    out_model_path = Path(out_model_path)
    ensure_dir(out_model_path.parent)
    joblib.dump({"rf": final_rf, "gb": final_gb, "features": FEATURE_NAMES}, str(out_model_path))

    return {
        "pearson_r": round(float(r_val), 3),
        "spearman_rho": round(float(rho_val), 3),
        "rmse": round(rmse_val, 3),
        "saved_path": str(out_model_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Train custom ML affinity model for ESM-AMP-Screen")
    parser.add_argument("--features-csv", help="Pre-extracted features CSV with 'experimental_pKd' column")
    parser.add_argument("--output-model", default="calibrated_peptide_model.pkl", help="Output .pkl model path")
    args = parser.parse_args()

    if args.features_csv and Path(args.features_csv).exists():
        df = pd.read_csv(args.features_csv)
        X = df[FEATURE_NAMES].values
        y = df["experimental_pKd"].values
        results = train_and_evaluate(X, y, Path(args.output_model))
        print("Training Complete:")
        print(f"  Pearson r:    {results['pearson_r']}")
        print(f"  Spearman rho: {results['spearman_rho']}")
        print(f"  RMSE:         {results['rmse']} pKd units")
        print(f"  Model saved:  {results['saved_path']}")
    else:
        print("Usage: python train_ml_model.py --features-csv features.csv --output-model calibrated_peptide_model.pkl")


if __name__ == "__main__":
    main()
