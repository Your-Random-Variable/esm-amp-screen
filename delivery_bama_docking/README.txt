==============================================================================
  ESM-AMP-Screen v1.5.0 — V. cholerae BamA Docking Delivery Package
==============================================================================
Date        : 2026-11-03
Platform    : ESM-AMP-Screen v1.5.0 + DeepSTAMP 650M ESM-2
Target      : BamA Outer Membrane Assembly Machine — Lateral Gate Seam
Receptor    : V. cholerae AlphaFold (UniProt Q9KPW0, 803 residues, AF v6)
Scaffold    : Plant Cyclotide (Cyclic Cystine Knot / CCK Fold, 31 aa)

------------------------------------------------------------------------------
FILE MANIFEST
------------------------------------------------------------------------------

PEPTIDE STRUCTURES (isolated, no receptor):
  WT_peptide.pdb            Wild-type cyclotide: DIFCGETCAFIPCITHVPGTCSCKSKVCYFN
  WT_peptide.pdbqt          Same — PDBQT format (Open Babel 3.1.0, Gasteiger charges)
  DesignB_peptide.pdb       Lead Design B:       KIFCGETCKFIPCITKVPGTCKCKSKVCYFN
  DesignB_peptide.pdbqt     Same — PDBQT format

RECEPTOR:
  BamA_VcholeraeAF_Q9KPW0_receptor.pdb     V. cholerae BamA AlphaFold structure
  BamA_VcholeraeAF_Q9KPW0_receptor.pdbqt   Same — PDBQT format

DOCKED COMPLEXES (receptor chain A + peptide chain B):
  WT_BamA_docked_complex.pdb               WT pose at BamA lateral gate
  WT_BamA_docked_complex.pdbqt             Same — PDBQT format
  DesignB_BamA_docked_complex.pdb          Design B pose at BamA lateral gate
  DesignB_BamA_docked_complex.pdbqt        Same — PDBQT format

------------------------------------------------------------------------------
DOCKING SUMMARY (V. cholerae BamA, Lateral Gate Pocket)
------------------------------------------------------------------------------
Pocket center : (4.88, 21.11, -0.39) Å
Box size      : 38 × 38 × 38 Å
Key residues  : Tyr432 (β1), Phe712/Gln720 (β16), Trp769/Val770/Leu780/Phe794 (L4/L6)

Variant       Charge  Dock (kcal/mol)  HADDOCK  Evdw    BSA (Å²)  Clashes  Consensus
WT (Baseline) -0.21   -7.09            67.72    81.47   1264.4    0        3.06*
Design B ★    +4.70   -8.43            45.93    60.39   1283.2    0        3.34

★ Design B = confirmed lead candidate
* WT has best ML ΔG but CANNOT cross LPS outer membrane (charge -0.21)

------------------------------------------------------------------------------
IMPORTANT NOTES
------------------------------------------------------------------------------
1. Peptide conformer: helix (ESM-AMP-Screen does not model disulfide bonds).
   For native cyclotide geometry, refer to HDOCK results (2MH1_A template).

2. HADDOCK scores are ESM-AMP-Screen internal scores (includes -0.05*BSA term
   REMOVED in v1.5.0). Not directly comparable to HADDOCK 2.4 web server.

3. ML Kd values are order-of-magnitude estimates (±5× uncertainty).

4. PDBQT files generated with Open Babel 3.1.0 using -xr (rigid) flag.
   Gasteiger partial charges assigned. Suitable for Vina/GNINA re-docking.

------------------------------------------------------------------------------
