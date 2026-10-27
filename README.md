# ESM-AMP-Screen

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Version-1.1.1-brightgreen" alt="Version 1.1.1">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Docking-AutoDock%20Vina%201.2-orange" alt="AutoDock Vina">
  <img src="https://img.shields.io/badge/Structure-ESMFold%20%7C%20PeptideBuilder-purple" alt="ESM & PeptideBuilder">
  <img src="https://img.shields.io/badge/Physics%20Engine-HADDOCK%202.4%20%7C%20AMBER%20ff14SB-red" alt="Physics & Force Fields">
  <img src="https://img.shields.io/badge/Tests-All%20Passing-brightgreen" alt="Tests">
</p>

**ESM-AMP-Screen** is a high-throughput, physics-grounded computational pipeline for **protein-peptide virtual screening, rigorous biophysical rescoring, and in silico lead optimization**. 

Designed for drug discovery teams, chemical biologists, and antimicrobial peptide (AMP) researchers, it bridges the gap between ultra-fast virtual screening (~10–25 seconds per peptide) and rigorous physical accuracy. It accurately models protein-peptide recognition, provides heuristic rank ordering of solution affinities ($K_d$), and engineers affinity-maturing substitutions.

---

## 🚀 Why ESM-AMP-Screen?

| Metric / Feature | Traditional Peptide Docking (Rosetta FlexPepDock, CrankPep, HADDOCK Web) | Standard Small-Molecule Docking (Vina, Glide, AutoDock 4) | **ESM-AMP-Screen** |
| :--- | :--- | :--- | :--- |
| **Speed per Peptide** | 15–120 minutes | 1–5 minutes | **10–25 seconds (100× faster)** |
| **Throughput** | 10–50 peptides / day | 500–1,000 / day (severe distortion on peptides) | **10,000+ peptides / day** |
| **Secondary Structure** | Heavy sampling needed | Shatters peptide helices/strands | **Preserved via Ramachandran dihedrals & ESM** |
| **Macrocycle Perception** | Manual topology scripting | Unconstrained | **Automatic SG-SG & cyclic topology locking** |
| **Charge Model** | All-atom with explicit H | Variable | **Robust united-atom formal charge mapping** |
| **Thermodynamic Output** | Gas-phase / empirical score | Raw rigid grid $\Delta G$ (overestimates by 2–5 kcal/mol) | **Entropy-penalized solution affinity ranking** |
| **In Silico Lead Opt** | Manual point mutagenesis | Not available | **Automated In Silico Alanine & Mutation Scan** |

---

## 🌟 3-Stage Screening Architecture

```
Peptide Library (FASTA / PDBs) ──► Target Protein (PDB / RCSB ID)
            │
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: High-Throughput Global Filter (Rigid ESM HTVS)                │
│ • Rapidly samples binding modes in seconds                             │
│ • Ranks by composite score (Vina ΔG, net charge, μH, salt bridges)    │
│ • Discards non-binders and identifies pocket localization              │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ (Top Candidates)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Induced-Fit Flexible Sidechain Refinement                     │
│ • Keeps backbone fixed in binding cleft                                │
│ • Flexes functional sidechains (Arg, Lys, Trp, Phe, Tyr, Asp, Glu)    │
│ • Resolves steric clashes and optimizes hydrogen-bonding networks      │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ (Top Refined Poses)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: HADDOCK Physics & Solution Thermodynamics Consensus          │
│ • United-atom charge model (cationic AMPs strictly positive formal Z)  │
│ • Continuous electrostatics (9–12 Å canonical C2 quintic switch)       │
│ • Lennard-Jones 12-6 with steep repulsive wall (+500 kcal/mol clash)   │
│ • Shrake–Rupley buried surface area (BSA) & Eisenberg desolvation      │
│ • Entropy-corrected solution affinity (ΔG_sol and solution Kd ranking) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Bound Co-Crystal Pose Rescoring & Sanity Check

The physics and solution thermodynamics engines are evaluated on **6 standardized benchmark co-crystal complexes** (shipped inside `benchmarks/`). This benchmark verifies physical force-field rescoring and rank-order correlation on known bound crystallographic structures—serving as an essential physical consistency sanity check across diverse peptide lengths (6–13 residues) and receptor classes, rather than a prospective blind docking validation. All structures have experimentally measured wet-lab dissociation constants ($K_d$ / $\Delta G_{\text{exp}}$) reported in the literature (*Science*, *Structure*, *Protein Science*):

| Benchmark PDB | Target Receptor | Peptide Sequence (Length) | $E_{\text{vdw}}$ (kcal/mol) | $E_{\text{elec}}$ (kcal/mol) | HADDOCK Score | BSA ($\text{\AA}^2$) | Predicted $\Delta G_{\text{sol}}$ | Experimental $\Delta G_{\text{exp}}$ | Predicted $K_d$ | Experimental $K_d$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1AWR** | Cyclophilin A (CypA) | `HAGPIA` (6 aa) | $-24.4$ | $-2.0$ | $-29.2$ | $780$ | **$-5.57\text{ kcal/mol}$** | **$-7.18\text{ kcal/mol}$** | $83.2\ \mu\text{M}$ | $5.5\ \mu\text{M}$ |
| **1CKA** | Abl SH3 Domain | `PPPALPPKK` (9 aa) | $-27.6$ | $-34.0$ | $-42.0$ | $938$ | **$-8.86\text{ kcal/mol}$** | **$-7.95\text{ kcal/mol}$** | $319.7\text{ nM}$ | $1.5\ \mu\text{M}$ |
| **1DKX** | DnaK Chaperone SBD | `NRLLLTG` (7 aa) | $-31.5$ | $-3.5$ | $-39.5$ | $1093$ | **$-7.13\text{ kcal/mol}$** | **$-8.73\text{ kcal/mol}$** | $5.9\ \mu\text{M}$ | $0.4\ \mu\text{M}$ |
| **1ER8** | Endothiapepsin | `HPFHLLVY` (8 aa) | $-61.2$ | $+0.1$ | $-72.7$ | $1616$ | **$-16.53\text{ kcal/mol}$** | **$-10.80\text{ kcal/mol}$** | $0.8\text{ pM}$ | $12\text{ nM}$ |
| **1SSH** | c-Src SH3 Domain | `GPPPAMPARPT` (11 aa) | $-29.7$ | $-15.5$ | $-39.7$ | $914$ | **$-7.70\text{ kcal/mol}$** | **$-6.95\text{ kcal/mol}$** | $2.3\ \mu\text{M}$ | $8.0\ \mu\text{M}$ |
| **1YCR** | MDM2 Oncoprotein | `ETFSDLWKLLPEN` (13 aa) | $-49.8$ | $-10.0$ | $-64.5$ | $1440$ | **$-11.85\text{ kcal/mol}$** | **$-8.49\text{ kcal/mol}$** | $2.1\text{ nM}$ | $600\text{ nM}$ |

### Statistical Validation Metrics:
- **Pearson Correlation ($r$):** **`0.867`** ($p = 0.0255$)
- **Spearman Rank Order ($\rho$):** **`0.543`**
- **Root Mean Square Error (RMSE):** **`2.91 kcal/mol`** across all 6 systems

> **100% Reproducible:** You can re-run and verify this entire benchmark table on your local workstation in ~10 seconds:
> ```bash
> python run_benchmarks.py
> # or via the CLI:
> esm-amp-screen benchmark --run-all
> ```

---

## ⚡ Multi-Threading & CPU Resource Management

ESM-AMP-Screen is built for high-throughput concurrency, scaling seamlessly from 4-core laptops to 128-core HPC compute nodes.

### Thread Budgeting Architecture (`--threads` / `-t` & `--max-workers`)

```
                          User Thread Budget: --threads T
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
     Concurrent Docking Workers             Search CPU Allocation per Worker
     W = min(threads, max_workers)          C_task = max(1, T // W)
```

1. **`--threads`, `-t` (Total CPU Thread Budget)**:
   - Sets the global upper bound on simultaneous CPU threads across all subprocesses.
   - Defaults to `os.cpu_count()` (all available CPU cores).
2. **`--max-workers` (Concurrent Task Processes)**:
   - Controls how many peptides dock simultaneously.
   - Defaults to `min(threads, 4)` to prevent RAM thrashing and disk I/O bottlenecks.
3. **Low-level Subprocess Allocation**:
   - Each individual AutoDock Vina process is given `threads // num_workers` CPU search threads.
   - **Strict upper bound:** Total active CPU usage will never exceed `--threads`.
4. **Lead Optimization Parallelism (`optimize-hit`)**:
   - The `--threads` parameter parallelizes alanine scanning and 20-amino-acid mutation scanning across a Python `ThreadPoolExecutor`, speeding up saturation mutagenesis from minutes to seconds.

### Recommended Settings:

| Environment | Cores | Recommended CLI Flags | Description |
| :--- | :--- | :--- | :--- |
| **Laptop / Desktop** | 4 – 8 | `--threads 6 --max-workers 2` | Smooth desktop responsiveness without overheating |
| **High-End Workstation** | 16 – 32 | `--threads 24 --max-workers 6` | High throughput (~150 peptides/hour) |
| **Compute Server / HPC** | 64 – 128 | `--threads 64 --max-workers 16` | Maximum batch throughput (~1,000+ peptides/hour) |

---

## 📦 Installation

### Method 1: Conda / Mamba (Recommended, 1-Command Setup)

```bash
# Clone the repository
git clone https://github.com/Your-Random-Variable/esm-amp-screen.git
cd esm-amp-screen

# Create and activate environment
conda env create -f environment.yml
conda activate esm-amp-screen
```

### Method 2: Pip & System Tools

Ensure `vina` (AutoDock Vina $\ge$ 1.2) and `obabel` (Open Babel $\ge$ 3.1) are available on your `PATH`:

```bash
# Ubuntu / Debian prerequisites (includes OpenBabel CLI & Python bindings)
sudo apt-get update && sudo apt-get install -y autodock-vina openbabel python3-openbabel

# Or with Conda:
conda install -c conda-forge autodock-vina openbabel

# Install ESM-AMP-Screen package
pip install -e .
```

> **Note on OpenBabel:** OpenBabel is used for PDBQT conversion and optional MMFF94 force-field interface relaxation in the lead optimizer. If OpenBabel Python bindings are absent, the pipeline automatically detects tool presence and gracefully scores poses directly without crashing.

Verify installation:
```bash
esm-amp-screen --version
```

---

## 📖 Complete CLI Options Reference

The `esm-amp-screen` CLI provides four powerful subcommands:
1. `run` (or `screen`): End-to-end automated 3-stage virtual screening pipeline
2. `optimize-hit`: In silico alanine scanning & affinity-maturing lead optimization
3. `score`: Standalone physics and solution thermodynamics rescoring for existing complexes
4. `benchmark`: CAPRI structural evaluation against reference crystal structures or standardized suites

---

### 1. `esm-amp-screen run` / `esm-amp-screen screen`

Runs the end-to-end multi-stage virtual screening funnel.

```bash
esm-amp-screen run \
  --receptor examples/receptors/target.pdb \
  --library examples/peptides/benchmark_amps.fasta \
  --output ./screening_results \
  --threads 8
```

#### Full Options Table:

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--receptor` | `-r` | Path / ID | *Required* | Receptor PDB file path or 4-letter RCSB PDB ID (e.g. `1YCR`, `2MLT`). |
| `--library` | `-l` | Path | *Required* | Peptide library FASTA file, or folder of pre-generated PDB structures. |
| `--output` | `-o` | Path | `./screening_results` | Output directory for all screening results, poses, and reports. |
| `--threads` | `-t` | Int | All CPUs | Total CPU threads allocated across parallel docking processes. |
| `--max-workers` | | Int | `min(threads, 4)` | Maximum concurrent docking worker processes. |
| `--center` | | Float × 3 | Auto | Grid box center coordinates `(x y z)` in Å. |
| `--size` | | Float × 3 | Auto | Grid box dimensions `(sx sy sz)` in Å. |
| `--pocket-residues`| | String | None | Comma-separated pocket residues (e.g. `'A:45,A:48,A:52'`). |
| `--auto-pocket` | | Flag | `True` | Automatically detect druggable surface cavities if `--center` is omitted. |
| `--no-auto-pocket`| | Flag | | Disable pocket detection; use whole-protein search box. |
| `--padding` | | Float | `8.0` | Grid box padding around receptor or detected pocket (Å). |
| `--conformation` | `-c` | Choice | `auto` | Secondary structure prior: `auto` (sequence propensity), `helix`, `sheet`, `extended`, `ppii`, or `ensemble`. |
| `--flexibility` | | Choice | `rigid` | Stage 1 flexibility: `rigid` (fast HTVS) or `sidechain`. |
| `--disulfide-lock`| | Flag | `True` | Perceive and lock intra-peptide disulfide bonds (SG-SG) and cyclic topology. |
| `--no-disulfide-lock` | | Flag | | Treat cysteines as standard flexible thiols. |
| `--exhaustiveness`| `-e` | Int | `2` | Stage 1 Vina search exhaustiveness. |
| `--refine-top` | | Int | `3` | Stage 2: Number of top hits to refine with flexible sidechains (set `0` to skip Stage 2). |
| `--refine-exhaustiveness` | | Int | `4` | Stage 2: Vina exhaustiveness for flexible sidechain refinement. |
| `--num-modes` | | Int | `9` | Maximum binding modes generated per peptide. |
| `--top` | | Int | `5` | Number of top candidates to export co-complex PDBs for. |
| `--hf-token` | | String | Env `HF_TOKEN` | HuggingFace API token for ESMFold cloud inference endpoint. |
| `--vina-bin` | | Path | From PATH | Custom path to AutoDock Vina binary. |
| `--obabel-bin` | | Path | From PATH | Custom path to OpenBabel binary. |
| `--ml-model` | | Path | None | Path to custom trained machine learning model checkpoint (`.pkl` / `.joblib`). |
| `--relax` | | Flag | `False` | Perform interface force-field relaxation (UFF) before Stage 3 physics scoring. |
| `--no-ml` | | Flag | `False` | Disable Stage 3 HADDOCK physics and consensus rescoring. |

> **Structure Generation & ESMFold:** By default, ESM-AMP-Screen builds 3D peptide conformers locally using its high-speed biophysical conformer engine with Ramachandran secondary-structure dihedrals (PeptideBuilder). To use a hosted ESMFold service, configure a custom private/dedicated endpoint via the `ESMFOLD_API_URL` environment variable and set `HF_TOKEN`.

---

### 2. `esm-amp-screen optimize-hit`

Performs alanine scanning to map hotspot residues and mutation scanning to engineer affinity gains.

```bash
esm-amp-screen optimize-hit \
  --complex screening_results/top_complexes/complex_Hit1.pdb \
  --rec-chain A \
  --pep-chain B \
  --mode all \
  --threads 8
```

#### Full Options Table:

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--complex` | `-c` | Path | *Required* | Input docked co-complex PDB file (receptor + peptide). |
| `--rec-chain` | | String | `A` | Receptor chain identifier in complex. |
| `--pep-chain` | | String | `B` | Peptide chain identifier in complex. |
| `--output` | `-o` | Path | `<cpx_dir>/lead_opt` | Output directory for optimization CSV reports and mutated PDBs. |
| `--mode` | | Choice | `all` | Optimization mode: `alanine` (hotspots), `rational` (mutations), or `all`. |
| `--positions` | | String | None | Comma-separated 1-based residue numbers to mutate (e.g. `'1,4,7'`). If omitted, scans tolerant interface positions. |
| `--substitutions` | | String | Rational | Comma-separated amino acids to test (e.g. `'W,F,Y,R,K,E,D'`). |
| `--pocket-residues`| | String | None | Target pocket residues for AIR restraint evaluation. |
| `--top-n` | | Int | `10` | Number of top optimized variants to export. |
| `--threads` | `-t` | Int | All CPUs | Number of CPU worker threads for parallel mutation scanning. |

> **Lead Optimization Scoring:** Predicted $\Delta\Delta G$ and fold-change values serve as empirical heuristic ranking metrics for prioritizing candidate substitutions in wet-lab assays, rather than all-atom thermodynamic free energies.

---

### 3. `esm-amp-screen score`

Standalone rescoring of any existing protein-peptide complex PDB using the full HADDOCK physics engine.

```bash
esm-amp-screen score \
  --complex benchmarks/1ycr.pdb \
  --rec-chain A \
  --pep-chain B \
  --relax
```

#### Full Options Table:

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--complex` | `-c` | Path | *Required* | Input co-complex PDB file to score. |
| `--rec-chain` | | String | `A` | Receptor chain identifier (default: `A`). |
| `--pep-chain` | | String | `B` | Peptide chain identifier (default: `B`). |
| `--pocket-residues`| | String | None | Comma-separated pocket residues for AIR restraint calculation. |
| `--relax` | | Flag | `False` | Perform interface energy minimization (50 steps UFF) before scoring. |
| `--output` | `-o` | Path | None | Optional directory to save relaxed complex structure and report. |

---

### 4. `esm-amp-screen benchmark`

Calculates CAPRI evaluation metrics against reference crystal structures or executes the validation suite.

```bash
# Option A: Run complete reproducible benchmark suite across 6 co-crystals
esm-amp-screen benchmark --run-all

# Option B: Evaluate a specific predicted complex against reference crystal
esm-amp-screen benchmark \
  --complex predicted_pose.pdb \
  --reference reference_crystal.pdb \
  --rec-chain A \
  --pep-chain B \
  --out-superimposed superimposed.pdb
```

#### Full Options Table:

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--run-all` | | Flag | `False` | Run standardized validation suite across all 6 benchmark co-crystals. |
| `--benchmarks-dir` | | Path | `./benchmarks` | Directory containing benchmark co-crystal PDB files. |
| `--complex` | `-c` | Path | None | Predicted / docked complex PDB file. |
| `--reference` | `-r` | Path | None | Reference experimental co-crystal PDB file. |
| `--rec-chain` | | String | `A` | Receptor chain ID in predicted complex. |
| `--pep-chain` | | String | `B` | Peptide chain ID in predicted complex. |
| `--ref-rec-chain`| | String | `--rec-chain` | Receptor chain ID in reference crystal. |
| `--ref-pep-chain`| | String | `--pep-chain` | Peptide chain ID in reference crystal. |
| `--out-superimposed` | `-o` | Path | None | Path to save receptor-superimposed complex PDB. |

---

## 📁 Output File Structure

A virtual screening run creates the following organized folder hierarchy:

```
screening_results/
├── screening_report.html             # Interactive HTML report (uses 3Dmol.js via cdnjs for WebGL rendering)
├── screening_leaderboard.csv         # Comprehensive multi-metric consensus leaderboard
├── screening_leaderboard_refined.csv # Stage 2 flexible sidechain refined candidate scores
├── screening_leaderboard_stage1.csv  # Stage 1 high-throughput screening scores
├── view_hits.pml                     # One-click PyMOL visualization script with color-coded contacts
├── target_receptor.pdb               # Prepared and cleaned receptor structure
├── top_complexes/                    # Combined receptor + peptide co-complex PDB structures
│   ├── complex_Hit1.pdb
│   └── complex_Hit2.pdb
├── docking_poses/                    # Stage 1 raw docked PDBQT poses and Vina logs
└── docking_poses_stage2/             # Stage 2 flexible sidechain refined poses
```

---

## 🧪 Testing & Verification

Run the comprehensive unit test, benchmark, and regression test suites:

```bash
# 1. Run Standardized Benchmark Validation Suite (bound co-crystal pose rescoring)
python run_benchmarks.py

# 2. Run Comprehensive Physics & Biochemical Audit Suite (18 verification checks)
python test_all_18_bugs.py

# 3. Run Automated Unit Test Suite (Unittest discovery)
python -m unittest discover -s tests

# 4. Run Pytest Suite
python -m pytest -q
```

---

## 📚 Citation

If you use **ESM-AMP-Screen** in your research, please cite:

```bibtex
@software{esm_amp_screen_2026,
  author = {ESM-AMP-Screen Computational Biology Team},
  title = {ESM-AMP-Screen: High-Throughput ESM-Powered Protein-Peptide Virtual Screening, Physics Engine, & In Silico Lead Optimizer},
  url = {https://github.com/Your-Random-Variable/esm-amp-screen},
  version = {1.1.1},
  year = {2026}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
