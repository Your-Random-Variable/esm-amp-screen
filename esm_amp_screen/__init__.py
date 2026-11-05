"""
AF-AMP-Screen / AMP-Screen: High-Throughput AlphaFold & Biophysics-Grounded Protein-Peptide Virtual Screening.
7-Ray PSP Multi-Pocket Blind Docking, Physics Engine, & In Silico Lead Optimizer.
"""

__version__ = "1.7.0"
__author__ = "Antigravity Computational Science Team"

from esm_amp_screen.alphafold import (
    fetch_alphafold_target,
    fetch_uniprot_sequence,
    prepare_alphafold_multimer_inputs,
    score_alphafold_complex
)
from esm_amp_screen.structure_builder import (
    build_biophysical_peptide,
    fold_peptide,
    batch_fold_peptides,
    predict_secondary_structure_propensity,
    fetch_rcsb_pdb
)

__all__ = [
    "__version__",
    "fetch_alphafold_target",
    "fetch_uniprot_sequence",
    "prepare_alphafold_multimer_inputs",
    "score_alphafold_complex",
    "build_biophysical_peptide",
    "fold_peptide",
    "batch_fold_peptides",
    "predict_secondary_structure_propensity",
    "fetch_rcsb_pdb",
]
