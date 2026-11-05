#!/usr/bin/env python3
"""
AlphaFold-Driven Virtual Screening Pipeline for Unknown Targets and Peptides.
=============================================================================
Replaces ESM with an end-to-end AlphaFold workflow:
1. Unknown Target: Fetches high-confidence predicted structure from AlphaFold Database (AFDB).
2. Unknown Peptides: Prepares AlphaFold-Multimer & AlphaFold 3 co-folding inputs.
3. Scoring & Triage: Evaluates co-folded complexes for interface pLDDT, HADDOCK physics,
   56D calibrated ML affinity (Kd, ΔG), and clinical toxicity (hemolysis & nephrotoxicity).

Usage:
    /home/lab202/miniconda3/envs/deepstamp_env/bin/python examples/run_alphafold_screening_pipeline.py
"""

from pathlib import Path
import pandas as pd

from esm_amp_screen.alphafold import (
    fetch_alphafold_target,
    prepare_alphafold_multimer_inputs,
    score_alphafold_complex
)
from esm_amp_screen.utils import ensure_dir

# 1. Target Definition (Unknown Target can be specified by UniProt Accession)
TARGET_UNIPROT = "Q9KPW0"  # Vibrio cholerae BamA Outer Membrane Assembly Factor
TARGET_NAME = "Vcholerae_BamA"

# 2. Designed Peptides (Unknown Peptides under evaluation)
AMP_VARIANTS = {
    "WT": {
        "sequence": "DIFCGETCAFIPCITHVPGTCSCKSKVCYFN",
        "description": "Natural Wild-Type Reference"
    },
    "Design_A": {
        "sequence": "KIFCGETCAFIPCITKVPGTCKCKSKVCYFN",
        "description": "Minimal 3-point mutant (D1K + H16K + S22K)"
    },
    "Design_B": {
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKSKVCYFN",
        "description": "Balanced 4-point mutant (D1K + A9K + H16K + S22K)"
    },
    "Design_C": {
        "sequence": "KIFCGETCKFIPCITKVPGTCKCKKKVCYFN",
        "description": "High-Cationic 5-point mutant"
    },
    "Design_D": {
        "sequence": "KIFCGKTCKFIPCITKVPGTCKCKKKVCYFN",
        "description": "Max-Score 6-point mutant"
    }
}


def main():
    base_out = Path("alphafold_screening_workspace")
    ensure_dir(base_out)

    print("\n" + "=" * 115)
    print("      ALPHAFOLD-DRIVEN VIRTUAL SCREENING PIPELINE FOR UNKNOWN TARGETS & PEPTIDES")
    print(f"      Target: {TARGET_NAME} (UniProt: {TARGET_UNIPROT}) | AlphaFold Database + AF-Multimer Suite")
    print("=" * 115 + "\n")

    # Step 1: Fetch target receptor structure from AlphaFold Database (AFDB)
    target_out_pdb = base_out / f"{TARGET_NAME}_AF.pdb"
    print(f"[Step 1] Fetching receptor structure from AlphaFold Database for UniProt: {TARGET_UNIPROT}...")
    target_info = fetch_alphafold_target(
        uniprot_id=TARGET_UNIPROT,
        out_pdb=target_out_pdb,
        clean=True
    )
    print(f"  ✓ Organism          : {target_info['organism']}")
    print(f"  ✓ Sequence Length   : {target_info['length']} residues")
    print(f"  ✓ Global pLDDT      : {target_info['global_plddt']:.1f} / 100")
    print(f"  ✓ High-Confidence   : {target_info['confident_residues_pct']:.1f}% residues (pLDDT ≥ 70)")
    print(f"  ✓ Receptor PDB      : {target_info['pdb_path']}\n")

    # Step 2: Prepare AlphaFold-Multimer & AlphaFold 3 Co-Folding Inputs
    print("[Step 2] Preparing AlphaFold-Multimer & AlphaFold 3 co-folding inputs for all candidates...")
    af_inputs_dir = base_out / "af_multimer_inputs"
    ensure_dir(af_inputs_dir)

    for pep_id, info in AMP_VARIANTS.items():
        pep_seq = info["sequence"]
        inputs = prepare_alphafold_multimer_inputs(
            target_seq=target_info["sequence"],
            peptide_seq=pep_seq,
            job_name=f"{TARGET_NAME}_{pep_id}",
            out_dir=af_inputs_dir / pep_id,
            target_name=TARGET_NAME,
            peptide_name=pep_id
        )
        print(f"  ✓ [{pep_id:<8}] FASTA: {inputs['colabfold_fasta'].name} | AF3 JSON: {inputs['af3_json'].name}")

    print("\n  [Tip] Inputs can be submitted to:")
    print("        • AlphaFold 3 Server: https://alphafoldserver.com (upload the _af3.json)")
    print("        • Google ColabFold: upload the _colabfold.fasta\n")

    # Step 3: Score and Triage Co-Folded Complexes
    print("[Step 3] Evaluating and scoring complex models with HADDOCK physics, 56D ML affinity, and toxicity...")
    results = []

    # Check for existing co-complex PDBs (e.g. from screening or AF-Multimer)
    delivery_dir = Path("delivery_bama_docking")
    for pep_id, info in AMP_VARIANTS.items():
        # Check delivery folder or screening results folder for complex structure
        cpx_path = delivery_dir / f"{pep_id}_BamA_docked_complex.pdb"
        if not cpx_path.exists():
            cpx_path = Path(f"screening_results_3target_v151/BamA/{pep_id}/complex_{pep_id}.pdb")

        if cpx_path.exists():
            score_res = score_alphafold_complex(
                complex_pdb=cpx_path,
                target_chain="A",
                peptide_chain="B"
            )
            score_res["Variant"] = pep_id
            score_res["Description"] = info["description"]
            results.append(score_res)

    if results:
        df = pd.DataFrame(results)
        csv_out = base_out / "alphafold_screening_decision_dashboard.csv"
        df.to_csv(csv_out, index=False)

        print("\n" + "=" * 125)
        print("                  ALPHAFOLD EXECUTIVE DECISION DASHBOARD & TRIAGE SUMMARY")
        print("=" * 125)
        hdr = f"{'Candidate':<10} {'Status':<14} {'Rating':<8} {'Net Chg':<8} {'ML ΔG':<8} {'Pred Kd':<9} {'BSA (Å²)':<9} {'Hemolysis':<14} {'Clinical Decision'}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            print(
                f"{r['Variant']:<10} {r['decision'][:12]:<14} {r['star_rating']:<8} "
                f"{r['net_charge']:<+8.1f} {r['ml_delta_g']:<8.2f} {r['ml_kd']:<9} "
                f"{r['bsa_angstrom2']:<9.1f} {r['hemolysis_risk'][:12]:<14} {r['decision']}"
            )
        print("=" * 125)
        print(f"\n[OK] Complete report saved to: {csv_out.resolve()}\n")


if __name__ == "__main__":
    main()
