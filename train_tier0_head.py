#!/usr/bin/env python3
"""
Tier-0 Sequence Model Training & Cross-Validation Engine for ESM-AMP-Screen.
Trains a calibrated classifier on 485-dimensional representations:
- 480-dimensional mean-pooled ESM-2 neural embeddings (facebook/esm2_t12_35M_UR50D)
- 5 Wimley-White biophysical membrane features:
  1. Eisenberg alpha-helical hydrophobic moment (100°)
  2. Eisenberg beta-sheet hydrophobic moment (160°)
  3. Wimley-White mean interfacial transfer free energy (ΔG_w->if, kcal/mol)
  4. Henderson-Hasselbalch charge density (Z / Length at pH 7.4)
  5. Membrane Permeabilization Index (MPI)

Evaluates 5-fold stratified cross-validation on 377 curated sequences:
- 197 diverse, experimentally validated active AMPs
- 180 rigorously matched negative controls (90 scrambled decoys + 90 natural non-AMPs)
Saves the final calibrated model to data/calibrated_tier0_head.joblib.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef
)

from esm_amp_screen.esm_embedder import ESM2Embedder
from esm_amp_screen.membrane_scorer import evaluate_membrane_activity
from esm_amp_screen.utils import logger, ensure_dir


def extract_features(sequences: list, embedder: ESM2Embedder, batch_size: int = 32) -> np.ndarray:
    """Extracts 485D representation for each sequence."""
    logger.info(f"Generating 480D ESM-2 embeddings for {len(sequences)} sequences...")
    esm_embs = embedder.batch_embed_sequences(sequences, batch_size=batch_size)
    
    logger.info("Computing Wimley-White interfacial and amphipathic moments...")
    feature_matrix = []
    for seq, emb in zip(sequences, esm_embs):
        mem = evaluate_membrane_activity(seq)
        biophys = np.array([
            mem["mu_h_alpha_100deg"],
            mem["mu_h_beta_160deg"],
            mem["mean_dg_interfacial_kcal"],
            mem["charge_density_z_per_res"],
            mem["membrane_permeabilization_index"]
        ])
        feat_vec = np.hstack([emb, biophys])
        feature_matrix.append(feat_vec)
        
    return np.array(feature_matrix)


def train_tier0_head(
    data_csv: Path = Path("data/full_training_set_sequence.csv"),
    out_model_path: Path = Path("data/calibrated_tier0_head.joblib"),
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    device: str = "cpu"
):
    print("=" * 105)
    print("      ESM-AMP-SCREEN: TIER-0 SEQUENCE CLASSIFIER TRAINING & CROSS-VALIDATION")
    print("=" * 105)
    
    data_csv = Path(data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_csv}")
        
    df = pd.read_csv(data_csv)
    n_total = len(df)
    n_pos = sum(df["label"] == 1)
    n_neg = sum(df["label"] == 0)
    print(f"Loaded dataset: {n_total} sequences ({n_pos} active AMPs, {n_neg} negative controls).")
    
    # Initialize ESM-2
    embedder = ESM2Embedder(model_name=model_name, device=device)
    embedder.load_model()
    
    # Feature extraction
    seq_list = df["sequence"].tolist()
    X = extract_features(seq_list, embedder=embedder)
    y = df["label"].values.astype(int)
    print(f"Feature matrix assembled: {X.shape[0]} samples × {X.shape[1]} dimensions.")
    
    # 5-Fold Stratified Cross-Validation
    folds = df["cv_fold"].values if "cv_fold" in df.columns else np.random.randint(0, 5, size=n_total)
    unique_folds = np.sort(np.unique(folds))
    
    y_pred_probs = np.zeros(n_total)
    
    print("\n--- Executing 5-Fold Cross-Validation ---")
    for fold in unique_folds:
        val_mask = (folds == fold)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        
        probs_va = clf.predict_proba(X_va)[:, 1]
        y_pred_probs[val_mask] = probs_va
        
        fold_auc = roc_auc_score(y_va, probs_va)
        print(f"  Fold {fold + 1}/5: {len(y_va)} validation samples | ROC-AUC: {fold_auc:.4f}")
        
    # Aggregate Cross-Validation Statistics
    cv_auc = roc_auc_score(y, y_pred_probs)
    prec, rec, _ = precision_recall_curve(y, y_pred_probs)
    cv_pr_auc = auc(rec, prec)
    
    y_pred_binary = (y_pred_probs >= 0.50).astype(int)
    acc = accuracy_score(y, y_pred_binary)
    bal_acc = balanced_accuracy_score(y, y_pred_binary)
    mcc = matthews_corrcoef(y, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_binary).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    # Top 20% Enrichment Factor
    k_20 = int(0.20 * n_total)
    top_20_idx = np.argsort(y_pred_probs)[::-1][:k_20]
    hits_in_top_20 = sum(y[top_20_idx])
    ef_20 = (hits_in_top_20 / k_20) / (n_pos / n_total)
    
    print("\n" + "=" * 105)
    print("      AGGREGATE 5-FOLD CROSS-VALIDATION SCREENING METRICS")
    print("=" * 105)
    print(f"  • Cross-Validated ROC-AUC:               {cv_auc:.4f}")
    print(f"  • Cross-Validated PR-AUC:               {cv_pr_auc:.4f}")
    print(f"  • Balanced Accuracy (threshold 0.50):    {bal_acc * 100.0:.2f}%")
    print(f"  • Sensitivity (True Positive Rate):     {sens * 100.0:.2f}% ({tp}/{tp+fn})")
    print(f"  • Specificity (Decoy Rejection Rate):   {spec * 100.0:.2f}% ({tn}/{tn+fp})")
    print(f"  • Matthews Correlation Coeff (MCC):     {mcc:.4f}")
    print(f"  • Enrichment Factor @ 20% (EF20%):       {ef_20:.2f}x (Max theoretical: {n_total/n_pos:.2f}x)")
    print("=" * 105)
    
    # Train final calibrated model on 100% of the training dataset
    print("\nTraining final calibrated model on all 377 sequences...")
    base_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=5)
    calibrated_clf.fit(X, y)
    
    out_model_path = Path(out_model_path)
    ensure_dir(out_model_path.parent)
    model_payload = {
        "model": calibrated_clf,
        "model_name": model_name,
        "n_samples": n_total,
        "cv_roc_auc": round(float(cv_auc), 4),
        "cv_pr_auc": round(float(cv_pr_auc), 4),
        "cv_specificity": round(float(spec), 4),
        "feature_dim": X.shape[1],
        "feature_names": [f"esm_{i}" for i in range(480)] + [
            "mu_h_alpha_100deg",
            "mu_h_beta_160deg",
            "mean_dg_interfacial_kcal",
            "charge_density_z_per_res",
            "membrane_permeabilization_index"
        ]
    }
    joblib.dump(model_payload, str(out_model_path))
    print(f"✔ Successfully saved calibrated model to: {out_model_path} ({out_model_path.stat().st_size} bytes)")
    
    return model_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tier-0 Sequence Model for ESM-AMP-Screen")
    parser.add_argument("--data", default="data/full_training_set_sequence.csv", help="Path to sequence training CSV")
    parser.add_argument("--output", default="data/calibrated_tier0_head.joblib", help="Output path for trained model")
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="ESM-2 model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    train_tier0_head(data_csv=args.data, out_model_path=args.output, model_name=args.model, device=args.device)
