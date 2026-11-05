"""
esm_amp_screen: High-Throughput ESM-Powered Protein-Peptide Virtual Screening, 7-Ray PSP Multi-Pocket Blind Docking, Physics Engine, & In Silico Lead Optimizer
"""

__version__ = "1.6.0"
__author__ = "Antigravity Science"

from esm_amp_screen.alphafold import (
    fetch_alphafold_target,
    fetch_uniprot_sequence,
    prepare_alphafold_multimer_inputs,
    score_alphafold_complex
)
