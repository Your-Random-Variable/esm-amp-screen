#!/usr/bin/env python3
"""
ESM-AMP-Screen Membrane Benchmark Suite.
Evaluates Tier-0 (ESM-2 Neural Embeddings) and Tier-1 (Wimley-White Membrane Biophysics)
on 30 curated DBAASP/APD3 experimentally validated antimicrobial peptides vs.
30 length- and composition-matched scrambled negative controls.

Calculates standard screening performance metrics:
- ROC-AUC
- PR-AUC (Average Precision)
- Enrichment Factor (EF10%, EF20%)
- Balanced Accuracy, Sensitivity, Specificity
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score, confusion_matrix

from esm_amp_screen.membrane_scorer import evaluate_membrane_activity
from esm_amp_screen.esm_embedder import ESM2Embedder
from esm_amp_screen.utils import logger

def run_membrane_benchmark(
    amp_csv: Path = Path("data/dbaasp_amp_benchmark.csv"),
    decoy_csv: Path = Path("data/matched_negative_decoys.csv"),
    out_csv: Path = Path("data/membrane_benchmark_results.csv"),
    use_esm: bool = True
):
    print("=" * 105)
    print("   ESM-AMP-SCREEN: MEMBRANE BIOPHYSICS & TIER-0 SEQUENCE BENCHMARK SUITE")
    print("=" * 105)
    
    if not amp_csv.exists() or not decoy_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV files missing: {amp_csv} or {decoy_csv}")
        
    df_amp = pd.read_csv(amp_csv)
    df_decoy = pd.read_csv(decoy_csv)
    
    print(f"Loaded {len(df_amp)} DBAASP/APD3 validated AMPs and {len(df_decoy)} matched scrambled negative decoys.")
    
    embedder = None
    if use_esm:
        print("Initializing local ESM-2 (facebook/esm2_t12_35M_UR50D)...")
        embedder = ESM2Embedder()
        embedder.load_model()
    
    records = []
    
    # 1. Process Active AMPs (Label = 1)
    for _, row in df_amp.iterrows():
        seq = row["sequence"]
        name = row["name"]
        mem = evaluate_membrane_activity(seq)
        esm_pred = embedder.predict_amp_probability(seq) if embedder else {"esm_prob_amp": 0.5, "is_uncertain_ood": False}
        
        # Composite Tier 0 + Tier 1 probability score
        comb_score = (0.50 * esm_pred["esm_prob_amp"]) + (0.50 * mem["bactericidal_probability"])
        
        records.append({
            "id": row["id"],
            "name": name,
            "sequence": seq,
            "length": row["length"],
            "label": 1,
            "group": "DBAASP_AMP",
            "net_charge_z": mem["net_charge_z"],
            "charge_density": mem["charge_density_z_per_res"],
            "dg_interfacial": mem["mean_dg_interfacial_kcal"],
            "mu_h_alpha": mem["mu_h_alpha_100deg"],
            "mu_h_beta": mem["mu_h_beta_160deg"],
            "mpi_score": mem["membrane_permeabilization_index"],
            "hemolysis_risk": mem["hemolysis_risk_probability"],
            "therapeutic_index": mem["therapeutic_index"],
            "esm_prob": esm_pred["esm_prob_amp"],
            "composite_score": round(comb_score, 4),
            "verdict": mem["verdict"],
            "experimental_mic": f"{row['mic_ug_ml']} µg/ml ({row['mic_umolar']} µM)"
        })
        
    # 2. Process Matched Decoys (Label = 0)
    for _, row in df_decoy.iterrows():
        seq = row["sequence"]
        name = row["matched_name"]
        mem = evaluate_membrane_activity(seq)
        esm_pred = embedder.predict_amp_probability(seq) if embedder else {"esm_prob_amp": 0.5, "is_uncertain_ood": False}
        
        comb_score = (0.50 * esm_pred["esm_prob_amp"]) + (0.50 * mem["bactericidal_probability"])
        
        records.append({
            "id": row["id"],
            "name": name,
            "sequence": seq,
            "length": row["length"],
            "label": 0,
            "group": "Scrambled_Decoy",
            "net_charge_z": mem["net_charge_z"],
            "charge_density": mem["charge_density_z_per_res"],
            "dg_interfacial": mem["mean_dg_interfacial_kcal"],
            "mu_h_alpha": mem["mu_h_alpha_100deg"],
            "mu_h_beta": mem["mu_h_beta_160deg"],
            "mpi_score": mem["membrane_permeabilization_index"],
            "hemolysis_risk": mem["hemolysis_risk_probability"],
            "therapeutic_index": mem["therapeutic_index"],
            "esm_prob": esm_pred["esm_prob_amp"],
            "composite_score": round(comb_score, 4),
            "verdict": mem["verdict"],
            "experimental_mic": "Inactive / Scrambled"
        })
        
    df_res = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    
    # 3. Compute Statistical Screening Metrics
    y_true = df_res["label"].values
    y_scores_mpi = df_res["mpi_score"].values / 100.0
    y_scores_esm = df_res["esm_prob"].values
    y_scores_comb = df_res["composite_score"].values
    
    roc_auc_mpi = roc_auc_score(y_true, y_scores_mpi)
    roc_auc_esm = roc_auc_score(y_true, y_scores_esm)
    roc_auc_comb = roc_auc_score(y_true, y_scores_comb)
    
    prec_comb, rec_comb, _ = precision_recall_curve(y_true, y_scores_comb)
    pr_auc_comb = auc(rec_comb, prec_comb)
    
    # Enrichment Factor at 20% selection
    n_total = len(df_res)
    n_pos = sum(y_true)
    k_20 = int(0.20 * n_total)
    top_20_idx = np.argsort(y_scores_comb)[::-1][:k_20]
    hits_in_top_20 = sum(y_true[top_20_idx])
    ef_20 = (hits_in_top_20 / k_20) / (n_pos / n_total)
    
    # Balanced Accuracy at threshold 0.5
    y_pred = (y_scores_comb >= 0.50).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    print("\n" + "="*105)
    print("      DISCRIMINATION METRICS (30 DBAASP AMPs vs. 30 Matched Decoys)")
    print("="*105)
    print(f"  • Wimley-White Membrane Biophysics Alone (Tier 1 ROC-AUC):   {roc_auc_mpi:.3f}")
    print(f"  • ESM-2 Sequence Embedding Model Alone (Tier 0 ROC-AUC):     {roc_auc_esm:.3f}")
    print(f"  • Unified Consensus (Tier 0 + Tier 1 ROC-AUC):               {roc_auc_comb:.3f}")
    print(f"  • Precision-Recall AUC (PR-AUC):                             {pr_auc_comb:.3f}")
    print(f"  • Enrichment Factor at Top 20% (EF20%):                      {ef_20:.2f}x (Max possible: 2.0x)")
    print(f"  • Balanced Accuracy (threshold 0.5):                         {bal_acc*100.0:.1f}%")
    print(f"  • Sensitivity (True Positive Rate):                          {sens*100.0:.1f}% ({tp}/{tp+fn})")
    print(f"  • Specificity (True Negative Rate):                          {spec*100.0:.1f}% ({tn}/{tn+fp})")
    print("="*105)
    
    # Sample Table Print
    print("\nSample Benchmark Predictions:")
    hdr = f"{'Name':<20} {'Class':<14} {'Charge':<8} {'ΔG_if':<8} {'μH(α)':<8} {'MPI':<7} {'HemoRisk':<10} {'CombProb':<10} {'MIC / Note':<20}"
    print(hdr)
    print("-" * len(hdr))
    for r in records[:6] + records[30:34]:
        print(
            f"{r['name'][:18]:<20} {r['group']:<14} {r['net_charge_z']:<8.1f} {r['dg_interfacial']:<8.2f} "
            f"{r['mu_h_alpha']:<8.2f} {r['mpi_score']:<7.1f} {r['hemolysis_risk']:<10.2f} {r['composite_score']:<10.3f} {r['experimental_mic'][:18]:<20}"
        )
    print("-" * len(hdr))
    print(f"\n[OK] Complete results saved to: {out_csv}\n")
    return df_res

if __name__ == "__main__":
    run_membrane_benchmark()
