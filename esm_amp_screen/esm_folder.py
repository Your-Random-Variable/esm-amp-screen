"""
Compatibility module: esm_folder.
=================================
Maintained for backwards compatibility after ESM removal.
All structure generation functions are now provided by structure_builder.py.
"""

from esm_amp_screen.structure_builder import (
    SECONDARY_STRUCTURE_DIHEDRALS,
    predict_secondary_structure_propensity,
    fetch_rcsb_pdb,
    build_biophysical_peptide,
    fold_peptide,
    batch_fold_peptides,
)

__all__ = [
    "SECONDARY_STRUCTURE_DIHEDRALS",
    "predict_secondary_structure_propensity",
    "fetch_rcsb_pdb",
    "build_biophysical_peptide",
    "fold_peptide",
    "batch_fold_peptides",
]
