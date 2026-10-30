"""
ESM-2 Neural Sequence Embedding and Representation Engine.

Integrates real, locally cached Meta AI ESM-2 protein language models:
- Default: facebook/esm2_t12_35M_UR50D (35M parameters, 480-dim embeddings, lightning-fast on CPU)
- Optional: facebook/esm2_t30_150M_UR50D (150M parameters, 640-dim embeddings)
Provides:
1. Exact sequence embeddings (mean-pooled over amino acid tokens)
2. Batch embedding inference with CPU thread parallelization
3. High-throughput representations for structural and downstream analysis
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from esm_amp_screen.utils import logger

# Default local/offline model tag (cached in ~/.cache/huggingface)
DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"


class ESM2Embedder:
    """
    High-Throughput ESM-2 Feature Extractor and Sequence Representation Engine.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._is_loaded = False

    def load_model(self):
        """Lazy loads the ESM-2 tokenizer and model on demand."""
        if self._is_loaded:
            return

        logger.info(f"Loading local ESM-2 model: {self.model_name} on {self.device}...")
        try:
            # Prefer locally cached weights without network calls
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info(f"ESM-2 model {self.model_name} loaded successfully (Hidden Dim: {self.model.config.hidden_size}).")
        except Exception as e:
            logger.error(f"Failed to load ESM-2 model {self.model_name}: {e}")
            raise RuntimeError(f"Could not initialize ESM-2: {e}")

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Generates mean-pooled hidden state representation for a single peptide sequence.
        Returns 1D numpy array of dimension D (e.g. 480 for 35M model).
        """
        sequence = sequence.strip().upper()
        if sequence in self._embedding_cache:
            return self._embedding_cache[sequence]

        self.load_model()

        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Exclude special tokens: [0] is BOS, [-1] is EOS
            last_hidden = outputs.last_hidden_state[0]
            if last_hidden.size(0) > 2:
                residue_reps = last_hidden[1:-1, :]
            else:
                residue_reps = last_hidden
            mean_rep = residue_reps.mean(dim=0).cpu().numpy()

        self._embedding_cache[sequence] = mean_rep
        return mean_rep

    def batch_embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Computes ESM-2 embeddings in batches for maximum throughput.
        """
        self.load_model()
        results = []
        uncached_seqs = []
        uncached_indices = []

        for i, seq in enumerate(sequences):
            s = seq.strip().upper()
            if s in self._embedding_cache:
                results.append((i, self._embedding_cache[s]))
            else:
                uncached_seqs.append(s)
                uncached_indices.append(i)

        for b_start in range(0, len(uncached_seqs), batch_size):
            b_seqs = uncached_seqs[b_start : b_start + batch_size]
            inputs = self.tokenizer(b_seqs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply attention mask for proper mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1)
                seq_lens = inputs["attention_mask"].sum(dim=1, keepdim=True)
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / torch.clamp(seq_lens, min=1.0)
                pooled_np = pooled.cpu().numpy()

            for j, p_emb in enumerate(pooled_np):
                seq_val = b_seqs[j]
                self._embedding_cache[seq_val] = p_emb
                global_idx = uncached_indices[b_start + j]
                results.append((global_idx, p_emb))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def analyze_sequence_embedding(
        self,
        sequence: str
    ) -> Dict[str, Any]:
        """
        Extracts ESM-2 neural sequence representations and embedding statistics.
        Provides 480-dimensional mean-pooled representations from ESM-2.
        """
        emb = self.embed_sequence(sequence)
        emb_norm = float(np.linalg.norm(emb))
        mean_act = float(np.mean(emb))
        var_act = float(np.var(emb))

        return {
            "sequence": sequence,
            "embedding": emb,
            "embedding_norm": round(emb_norm, 3),
            "mean_activation": round(mean_act, 4),
            "variance": round(var_act, 4),
            "embedding_dim": len(emb)
        }

    def predict_amp_probability(
        self,
        sequence: str,
        calibrated_model_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Compatibility method returning sequence embedding properties.
        Binary classification has been removed in favor of physical biophysical profiling.
        """
        analysis = self.analyze_sequence_embedding(sequence)
        return {
            "sequence": sequence,
            "embedding_norm": analysis["embedding_norm"],
            "embedding_dim": analysis["embedding_dim"],
            "mean_activation": analysis["mean_activation"]
        }
