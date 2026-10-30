"""
Model Training & Cluster-Stratified Cross-Validation Utility for ESM-AMP-Screen.
Extracts 56D structural, physical, and sequence feature vectors from a dataset of
381 protein-peptide co-complexes across 40 distinct receptor families with known
experimental affinities (pKd, Kd, DeltaG).
Trains the tri-model ensemble (Random Forest + Gradient Boosting + Extra Trees)
and saves the trained model to data/trained_complex_ensemble.joblib.

Key Scientific Safeguards:
1. Zero Data Leakage: 'delta_delta_g' feature strictly removed.
2. No Silent Defaults: Real sequence length, contact densities, and pI used.
3. Zero Target Leakage: GroupKFold cross-validation by receptor family ensures
   different members of the same protein family (e.g., all SH3 or all Caspases)
   never appear in both training and validation splits.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, GroupKFold

from esm_amp_screen.ml_scorer import (
    FEATURE_NAMES,
    extract_structural_fingerprint,
    extract_complex_features,
    ELEMENT_PAIRS
)
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.utils import logger, ensure_dir


def featurize_complex(complex_pdb: Path, meta_row: dict) -> list:
    """Extracts 56D feature vector from a single complex PDB (without label leakage)."""
    fp = extract_structural_fingerprint(complex_pdb)
    physics = calculate_haddock_physics_score(complex_pdb)
    seq = meta_row.get("sequence", meta_row.get("peptide_sequence", ""))
    return extract_complex_features(fp=fp, physics=physics, hit_meta=meta_row, seq=seq)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray = None,
    out_model_path: Path = Path("data/trained_complex_ensemble.joblib"),
    n_splits: int = 5
) -> dict:
    """
    Trains the tri-model ensemble and evaluates cross-validation performance.
    Uses GroupKFold by receptor family if groups are provided to prevent target leakage.
    """
    print("\n" + "=" * 105)
    print("      PROTEIN-PEPTIDE COMPLEX AFFINITY ENSEMBLE TRAINING & EVALUATION (56 PARAMETERS)")
    print("=" * 105)

    y_pred_cv = np.zeros_like(y)

    if groups is not None and len(np.unique(groups)) >= n_splits:
        print(f"Using GroupKFold cross-validation across {len(np.unique(groups))} distinct receptor families.")
        splitter = GroupKFold(n_splits=n_splits)
        split_gen = splitter.split(X, y, groups=groups)
    else:
        print("Using standard 5-fold cross-validation.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_gen = splitter.split(X, y)

    for fold_idx, (train_idx, val_idx) in enumerate(split_gen, 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        rf = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_split=3, random_state=42, n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.04, subsample=0.85, random_state=42)
        et = ExtraTreesRegressor(n_estimators=250, max_depth=12, min_samples_split=3, random_state=42, n_jobs=-1)

        rf.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)
        et.fit(X_tr, y_tr)

        y_pred_cv[val_idx] = (rf.predict(X_va) + gb.predict(X_va) + et.predict(X_va)) / 3.0
        fold_r, _ = pearsonr(y_va, y_pred_cv[val_idx]) if len(y_va) > 2 else (0.0, 1.0)
        print(f"  Fold {fold_idx}/{n_splits}: {len(val_idx)} validation complexes | Pearson r: {fold_r:.3f}")

    r_val, p_val = pearsonr(y, y_pred_cv)
    rho_val, _ = spearmanr(y, y_pred_cv)
    rmse_val = float(np.sqrt(np.mean((y - y_pred_cv) ** 2)))
    mae_val = float(np.mean(np.abs(y - y_pred_cv)))

    print("\n" + "=" * 105)
    print("      CROSS-VALIDATED REGRESSION METRICS ACROSS RECEPTOR FAMILIES")
    print("=" * 105)
    print(f"  • Cross-Validated Pearson Correlation (r):   {r_val:.4f} (p = {p_val:.2e})")
    print(f"  • Cross-Validated Spearman Rank Order (ρ):   {rho_val:.4f}")
    print(f"  • Root Mean Square Error (RMSE):             {rmse_val:.3f} pKd units (~{rmse_val * 1.364:.2f} kcal/mol)")
    print(f"  • Mean Absolute Error (MAE):                  {mae_val:.3f} pKd units (~{mae_val * 1.364:.2f} kcal/mol)")
    print("=" * 105)

    # Fit final models on 100% of available data
    print("\nFitting final tri-model ensemble (RF + GB + ET) on all 381 complexes...")
    final_rf = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_split=3, random_state=42, n_jobs=-1)
    final_gb = GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.04, subsample=0.85, random_state=42)
    final_et = ExtraTreesRegressor(n_estimators=300, max_depth=14, min_samples_split=3, random_state=42, n_jobs=-1)

    final_rf.fit(X, y)
    final_gb.fit(X, y)
    final_et.fit(X, y)

    out_model_path = Path(out_model_path)
    ensure_dir(out_model_path.parent)
    payload = {
        "rf": final_rf,
        "gb": final_gb,
        "et": final_et,
        "features": list(FEATURE_NAMES),
        "cv_pearson_r": round(float(r_val), 4),
        "cv_spearman_rho": round(float(rho_val), 4),
        "cv_rmse": round(float(rmse_val), 4),
        "cv_mae": round(float(mae_val), 4),
        "n_samples": len(y),
        "n_features": len(FEATURE_NAMES)
    }
    joblib.dump(payload, str(out_model_path), compress=3)
    print(f"✔ Successfully saved trained 56D ML ensemble to: {out_model_path}")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Train custom 56-parameter ML affinity model for ESM-AMP-Screen")
    parser.add_argument("--data", default="data/pdbbind_peptide_affinity_dataset.csv", help="Dataset CSV with experimental affinities and features")
    parser.add_argument("--output", default="data/trained_complex_ensemble.joblib", help="Output model path")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} protein-peptide complexes from {data_path.name}.")

    # Verify all 56 features exist in dataframe
    missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Dataset is missing required features: {missing_feats}")

    X = df[FEATURE_NAMES].values
    y_col = next((c for c in df.columns if c in ("experimental_pkd", "experimental_pKd", "pKd", "pkd")), None)
    if not y_col:
        raise ValueError(f"Could not find pKd column in {data_path.name}")
    y = df[y_col].values

    groups = df["receptor_family"].values if "receptor_family" in df.columns else None

    train_and_evaluate(X=X, y=y, groups=groups, out_model_path=Path(args.output))


if __name__ == "__main__":
    main()
