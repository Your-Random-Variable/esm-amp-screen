"""
Utility functions for file I/O, sequence parsing, and environment verification.
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("esm_amp_screen")

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "DHI": "H", "HIP": "H", "HIE": "H", "HID": "H",
    "CYX": "C", "SEC": "C", "PYL": "K", "NLE": "L", "NVA": "V"
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items() if len(k) == 3}

NON_STANDARD_MAP = {
    "U": "C",  # Selenocysteine -> Cys
    "O": "K",  # Pyrrolysine -> Lys
    "B": "D",  # Asp/Asn ambiguity -> Asp
    "Z": "E",  # Glu/Gln ambiguity -> Glu
    "J": "L",  # Leu/Ile ambiguity -> Leu
}

def ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def which_tool(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        # Also check current python's bin directory (e.g. conda env or venv)
        prefix_bin = Path(sys.prefix) / "bin" / cmd
        if prefix_bin.exists() and prefix_bin.is_file():
            return str(prefix_bin)
        raise RuntimeError(f"Required executable '{cmd}' not found in PATH.")
    return path

def clean_sequence(seq: str, allow_unknown: bool = True) -> str:
    """
    Strip whitespace, numbers, map non-standard amino acids, and validate.
    Maps IUPAC non-standard residues with biochemical equivalents:
      U -> C (Sec), O -> K (Pyl), B -> D (Asx), Z -> E (Glx), J -> L (Xle).
    If allow_unknown=False (strict mode), raises ValueError upon encountering 'X'
    or any unknown character. If allow_unknown=True, substitutes an alanine stub ('A')
    and emits an explicit warning to ensure provenance tracking.
    """
    raw_seq = "".join(seq.split()).upper()
    seq_alpha = "".join([c for c in raw_seq if c.isalpha()])
    cleaned = []
    for aa in seq_alpha:
        if aa in VALID_AMINO_ACIDS:
            cleaned.append(aa)
        elif aa in NON_STANDARD_MAP:
            mapped = NON_STANDARD_MAP[aa]
            logger.info(f"Mapping non-standard amino acid '{aa}' -> '{mapped}'.")
            cleaned.append(mapped)
        elif allow_unknown:
            logger.warning(
                f"Unresolved unknown amino acid residue '{aa}' in sequence '{raw_seq}'. "
                "Substituting Alanine ('A') stub for 3D modeling; verify sequence provenance."
            )
            cleaned.append("A")
        else:
            raise ValueError(
                f"Unresolved non-standard or unknown amino acid '{aa}' in peptide sequence '{raw_seq}'. "
                "Set allow_unknown=True to substitute Alanine ('A') stub."
            )
    return "".join(cleaned)

def read_fasta(fasta_path: Path) -> Dict[str, str]:
    """Parse a FASTA file into a dictionary of {id: sequence}."""
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    records = {}
    current_id = None
    seq_chunks = []
    
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    clean_seq = clean_sequence("".join(seq_chunks))
                    if clean_seq:
                        records[current_id] = clean_seq
                current_id = line[1:].split()[0]  # First token of header
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if current_id:
            clean_seq = clean_sequence("".join(seq_chunks))
            if clean_seq:
                records[current_id] = clean_seq
                
    logger.info(f"Loaded {len(records)} peptide sequences from {fasta_path.name}")
    return records
