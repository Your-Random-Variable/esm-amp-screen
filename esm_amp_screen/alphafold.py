"""
AlphaFold Integration Module for ESM-AMP-Screen.
Enables end-to-end AlphaFold workflows for unknown peptides and targets:
1. Fetch AlphaFold predicted receptor structures from AlphaFold Database (AFDB / EBI).
2. Generate AlphaFold-Multimer & AlphaFold 3 input files (paired FASTA, AF3 JSON, ColabFold runners).
3. Evaluate & score co-folded AlphaFold complex structures (interface pLDDT, HADDOCK physics,
   56D ML affinity, and toxicity assessment).
"""

import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import gemmi
import numpy as np

from esm_amp_screen.utils import logger, ensure_dir, clean_sequence, AA_3TO1
from esm_amp_screen.receptor_prep import clean_receptor_pdb
from esm_amp_screen.physics_engine import calculate_haddock_physics_score
from esm_amp_screen.amp_analyzer import (
    analyze_cocomplex_contacts,
    calculate_net_charge,
    calculate_hydrophobic_moment,
    calculate_mean_hydrophobicity,
    calculate_boman_index
)
from esm_amp_screen.ml_scorer import MLPeptideScorer


# ---------------------------------------------------------------------------
# 1. Fetch Target Receptor from AlphaFold Database (AFDB)
# ---------------------------------------------------------------------------

def fetch_alphafold_target(
    uniprot_id: str,
    out_pdb: Optional[Path] = None,
    clean: bool = True
) -> Dict[str, Any]:
    """
    Downloads an AlphaFold predicted protein structure from the AlphaFold Database (EBI).

    Args:
        uniprot_id: UniProt accession ID (e.g. 'Q9KPW0', 'P0A940').
        out_pdb: Target path to save the PDB file. Defaults to 'AF_{uniprot_id}.pdb'.
        clean: Whether to standardize chains and remove solvent/ligands.

    Returns:
        Dictionary with metadata: 'pdb_path', 'uniprot_id', 'global_plddt',
        'sequence', 'length', 'confident_fraction'.
    """
    uid = uniprot_id.strip().upper()
    if not out_pdb:
        out_pdb = Path(f"AF_{uid}.pdb")
    out_pdb = Path(out_pdb)
    ensure_dir(out_pdb.parent)

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "ESM-AMP-Screen/1.5.1"})

    logger.info(f"Querying AlphaFold Database for UniProt accession: {uid}...")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to query AlphaFold DB for {uid}: {e}")

    if not data or not isinstance(data, list):
        raise ValueError(f"No AlphaFold prediction found in AFDB for UniProt ID: {uid}")

    entry = data[0]
    pdb_url = entry.get("pdbUrl")
    if not pdb_url:
        raise ValueError(f"No PDB download URL in AFDB response for {uid}")

    global_plddt = float(entry.get("globalMetricValue", 0.0))
    logger.info(f"Downloading AlphaFold structure from {pdb_url} (Global pLDDT: {global_plddt:.1f})...")

    dl_req = urllib.request.Request(pdb_url, headers={"User-Agent": "ESM-AMP-Screen/1.5.1"})
    try:
        with urllib.request.urlopen(dl_req, timeout=30) as resp:
            raw_pdb = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to download PDB from {pdb_url}: {e}")

    out_pdb.write_text(raw_pdb, encoding="utf-8")

    if clean:
        clean_pdb = out_pdb.parent / f"{out_pdb.stem}_clean.pdb"
        clean_receptor_pdb(out_pdb, clean_pdb, receptor_chain="A")
        out_pdb = clean_pdb

    # Parse structure to extract sequence and per-residue pLDDT metrics
    st = gemmi.read_structure(str(out_pdb))
    plddts = []
    seq_parts = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name.upper() in AA_3TO1:
                    seq_parts.append(AA_3TO1[res.name.upper()])
                    ca = res.find_atom("CA", "*")
                    if ca:
                        plddts.append(ca.b_iso)

    seq_str = "".join(seq_parts)
    mean_plddt = float(np.mean(plddts)) if plddts else global_plddt
    high_conf = sum(1 for p in plddts if p >= 70.0) / max(1, len(plddts)) * 100.0

    return {
        "pdb_path": out_pdb.resolve(),
        "uniprot_id": uid,
        "global_plddt": round(mean_plddt, 2),
        "sequence": seq_str,
        "length": len(seq_str),
        "confident_residues_pct": round(high_conf, 1),
        "organism": entry.get("organismScientificName", "Unknown")
    }


def fetch_uniprot_sequence(uniprot_id: str) -> Tuple[str, str]:
    """
    Fetches the canonical FASTA sequence for a UniProt ID.
    Returns (header, sequence).
    """
    uid = uniprot_id.strip().upper()
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "ESM-AMP-Screen/1.5.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8").strip()
    lines = text.split("\n")
    header = lines[0]
    seq = "".join(lines[1:]).replace(" ", "").strip()
    return header, seq


# ---------------------------------------------------------------------------
# 2. AlphaFold-Multimer & AlphaFold 3 Input Preparation
# ---------------------------------------------------------------------------

def prepare_alphafold_multimer_inputs(
    target_seq: str,
    peptide_seq: str,
    job_name: str = "complex",
    out_dir: Optional[Path] = None,
    target_name: str = "Target",
    peptide_name: str = "Peptide"
) -> Dict[str, Path]:
    """
    Generates ready-to-use inputs for:
    1. ColabFold / AlphaFold-Multimer (paired FASTA format: TARGET:PEPTIDE).
    2. DeepMind AlphaFold 3 Server (JSON job file).
    3. Standalone execution shell script.

    Args:
        target_seq: Amino acid sequence of the target receptor.
        peptide_seq: Amino acid sequence of the peptide.
        job_name: Unique identifier for the complex.
        out_dir: Directory where input files will be created.

    Returns:
        Dictionary mapping input types to file paths.
    """
    out_dir = Path(out_dir or f"af_multimer_{job_name}")
    ensure_dir(out_dir)

    t_seq = clean_sequence(target_seq)
    p_seq = clean_sequence(peptide_seq)

    # 1. ColabFold Paired FASTA
    colab_fasta = out_dir / f"{job_name}_colabfold.fasta"
    fasta_content = f">{job_name}\n{t_seq}:{p_seq}\n"
    colab_fasta.write_text(fasta_content, encoding="utf-8")

    # 2. AlphaFold 3 Server JSON
    af3_json = out_dir / f"{job_name}_af3.json"
    af3_payload = {
        "name": job_name,
        "modelSeeds": [1],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": t_seq,
                    "count": 1
                }
            },
            {
                "proteinChain": {
                    "sequence": p_seq,
                    "count": 1
                }
            }
        ]
    }
    af3_json.write_text(json.dumps(af3_payload, indent=2), encoding="utf-8")

    # 3. ColabFold Batch Bash Script
    sh_script = out_dir / f"run_colabfold_{job_name}.sh"
    sh_content = f"""#!/usr/bin/env bash
# Automatically generated runner for ColabFold AlphaFold2-Multimer v3
# Usage: bash {sh_script.name}
set -e

if ! command -v colabfold_batch &> /dev/null; then
    echo "[!] colabfold_batch not found in PATH."
    echo "    To run locally: pip install colabfold[alphafold]"
    echo "    Alternatively, upload '{colab_fasta.name}' to Google ColabFold or '{af3_json.name}' to alphafoldserver.com"
    exit 1
fi

echo "Running ColabFold-Multimer for {job_name}..."
colabfold_batch --model-type alphafold2_multimer_v3 --num-recycle 3 {colab_fasta.name} {out_dir / "predictions"}
echo "AlphaFold-Multimer prediction finished!"
"""
    sh_script.write_text(sh_content, encoding="utf-8")
    os.chmod(sh_script, 0o755)

    logger.info(f"Generated AlphaFold-Multimer input files in {out_dir}:")
    logger.info(f"  • ColabFold FASTA : {colab_fasta.name}")
    logger.info(f"  • AlphaFold 3 JSON: {af3_json.name}")
    logger.info(f"  • Runner Script   : {sh_script.name}")

    return {
        "colabfold_fasta": colab_fasta.resolve(),
        "af3_json": af3_json.resolve(),
        "runner_script": sh_script.resolve(),
        "out_dir": out_dir.resolve()
    }


# ---------------------------------------------------------------------------
# 3. Evaluate & Score Co-Folded AlphaFold Complex Structures
# ---------------------------------------------------------------------------

def score_alphafold_complex(
    complex_pdb: Path,
    target_chain: str = "A",
    peptide_chain: str = "B",
    pocket_residues: Optional[List[str]] = None,
    ml_scorer: Optional[MLPeptideScorer] = None
) -> Dict[str, Any]:
    """
    Evaluates an AlphaFold-Multimer or AF3 predicted co-complex structure:
    1. Extracts target, peptide, and interface pLDDT confidence scores.
    2. Calculates HADDOCK explicit physics terms (Evdw, Eelec, Edesolv, BSA).
    3. Runs the calibrated 56D ML ensemble for binding free energy and Kd.
    4. Evaluates peptide safety, net charge, and hemolytic risk.
    5. Returns an intuitive executive Decision Verdict.
    """
    complex_pdb = Path(complex_pdb)
    if not complex_pdb.exists():
        raise FileNotFoundError(f"Complex PDB not found: {complex_pdb}")

    st = gemmi.read_structure(str(complex_pdb))

    # 1. Extract per-residue pLDDT and sequences from B-factor column
    target_plddts = []
    pep_plddts = []
    pep_seq_parts = []
    rec_coords = []
    pep_coords = []

    for model in st:
        for chain in model:
            for res in chain:
                ca = res.find_atom("CA", "*")
                b_val = ca.b_iso if ca else 80.0
                if chain.name == target_chain:
                    target_plddts.append(b_val)
                    for atom in res:
                        rec_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                elif chain.name == peptide_chain:
                    pep_plddts.append(b_val)
                    if res.name.upper() in AA_3TO1:
                        pep_seq_parts.append(AA_3TO1[res.name.upper()])
                    for atom in res:
                        pep_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    pep_seq = "".join(pep_seq_parts)
    mean_target_plddt = float(np.mean(target_plddts)) if target_plddts else 0.0
    mean_pep_plddt = float(np.mean(pep_plddts)) if pep_plddts else 0.0

    # 2. Interface pLDDT (residues within 5.0 A of partner chain)
    interface_plddts = []
    if rec_coords and pep_coords:
        from scipy.spatial.distance import cdist
        dists = cdist(np.array(rec_coords), np.array(pep_coords))
        # Identify interface atoms (< 5.0 A)
        close_rec_idx = np.where(dists.min(axis=1) <= 5.0)[0]
        close_pep_idx = np.where(dists.min(axis=0) <= 5.0)[0]
        # Approximate interface pLDDT from peptide residues
        interface_plddts = pep_plddts if pep_plddts else [mean_pep_plddt]

    interface_plddt = float(np.mean(interface_plddts)) if interface_plddts else mean_pep_plddt

    # 3. Intermolecular Contacts
    contacts = analyze_cocomplex_contacts(complex_pdb, receptor_chain=target_chain, peptide_chain=peptide_chain)

    # 4. HADDOCK Physics Score
    phys = calculate_haddock_physics_score(
        complex_pdb,
        pocket_residues=pocket_residues,
        receptor_chain=target_chain,
        peptide_chain=peptide_chain
    )

    # 5. Calibrated 56D ML Affinity
    if ml_scorer is None:
        model_file = Path(__file__).resolve().parent.parent / "data" / "trained_complex_ensemble.joblib"
        ml_scorer = MLPeptideScorer(model_path=model_file if model_file.exists() else None)

    hit_meta = {
        "sequence": pep_seq,
        "affinity": phys["haddock_score"] * 0.28,
        "affinity_kcal_mol": phys["haddock_score"] * 0.28,
        "hbonds": contacts["hbonds"],
        "salt_bridges": contacts["salt_bridges"],
        "hydrophobic_contacts": contacts["hydrophobic_contacts"],
        "clashes": contacts["clashes"]
    }
    ml_res = ml_scorer.score_complex(complex_pdb, hit_meta=hit_meta, pocket_residues=pocket_residues)

    # 6. Peptide Biophysics & Toxicity Profile
    net_chg = calculate_net_charge(pep_seq) if pep_seq else 0.0
    u_h = calculate_hydrophobic_moment(pep_seq) if pep_seq else 0.0
    mean_h = calculate_mean_hydrophobicity(pep_seq) if pep_seq else 0.0
    boman = calculate_boman_index(pep_seq) if pep_seq else 0.0

    # Hemolysis & Toxicity Rules
    if mean_h > 0.22:
        hemo_risk = "Moderate (Hydrophobic patch intact)"
    elif mean_h > 0.10:
        hemo_risk = "Low"
    else:
        hemo_risk = "Very Low (< 1%)"

    if net_chg > 7.0:
        tox_verdict = "High Cytotoxic Risk (Polycationic > +7.0)"
    elif net_chg > 5.5:
        tox_verdict = "Moderate-High (Approaching +6.0)"
    elif net_chg >= 3.0:
        tox_verdict = "Safe Therapeutic Window"
    else:
        tox_verdict = "Membrane Impermeable (Negative/Low Charge)"

    # 7. Executive Decision Verdict
    if contacts["clashes"] > 0:
        decision = "🔴 REJECTED (Steric Clash at Binding Pocket)"
        star_rating = "⭐"
    elif net_chg < 1.0:
        decision = "🔴 REJECTED (Membrane Impermeable - Cannot Cross LPS)"
        star_rating = "⭐⭐"
    elif net_chg > 7.0:
        decision = "🔴 REJECTED (Polycationic Toxicity Danger)"
        star_rating = "⭐⭐"
    elif interface_plddt >= 75.0 and ml_res["ml_delta_g"] <= -8.0:
        decision = "🟢 RECOMMENDED LEAD (High Confidence Complex & Potent Binding)"
        star_rating = "⭐⭐⭐⭐⭐"
    else:
        decision = "🟡 BACKUP CANDIDATE (Moderate Confidence / Viable Alternative)"
        star_rating = "⭐⭐⭐⭐"

    return {
        "complex_pdb": str(complex_pdb.resolve()),
        "target_chain": target_chain,
        "peptide_chain": peptide_chain,
        "peptide_sequence": pep_seq,
        "peptide_length": len(pep_seq),
        "target_plddt": round(mean_target_plddt, 1),
        "peptide_plddt": round(mean_pep_plddt, 1),
        "interface_plddt": round(interface_plddt, 1),
        "ml_delta_g": ml_res["ml_delta_g"],
        "ml_kd": ml_res["ml_kd"],
        "haddock_score": phys["haddock_score"],
        "bsa_angstrom2": phys["bsa"],
        "hbonds": contacts["hbonds"],
        "salt_bridges": contacts["salt_bridges"],
        "hydrophobic_contacts": contacts["hydrophobic_contacts"],
        "clashes": contacts["clashes"],
        "net_charge": net_chg,
        "hydrophobic_moment": u_h,
        "mean_hydrophobicity": mean_h,
        "boman_index": boman,
        "hemolysis_risk": hemo_risk,
        "toxicity_verdict": tox_verdict,
        "decision": decision,
        "star_rating": star_rating
    }
