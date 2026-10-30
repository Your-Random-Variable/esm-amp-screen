"""
Unit Tests for ESM-AMP-Screen Membrane Biophysics and ESM-2 Embedding Modules.
Tests Wimley-White thermodynamics, Eisenberg amphipathic moments,
Henderson-Hasselbalch charge calculations, and ESM-2 neural inference.
"""

import unittest
import numpy as np
from pathlib import Path

from esm_amp_screen.membrane_scorer import (
    WIMLEY_WHITE_INTERFACIAL,
    WIMLEY_WHITE_OCTANOL,
    EISENBERG_CONSENSUS,
    clean_amp_sequence,
    calculate_net_charge_at_ph,
    calculate_wimley_white_interfacial_dg,
    calculate_wimley_white_octanol_dg,
    calculate_amphipathic_moment,
    evaluate_membrane_activity
)
from esm_amp_screen.esm_embedder import ESM2Embedder


class TestMembraneBiophysics(unittest.TestCase):
    """Verifies Wimley-White thermodynamic parameters and membrane calculations."""

    def test_wimley_white_scales(self):
        """Verify experimental reference values for Wimley-White interfacial and octanol scales."""
        # Trp and Phe must be strongly favorable for interfacial partitioning
        self.assertEqual(WIMLEY_WHITE_INTERFACIAL["W"], -1.85)
        self.assertEqual(WIMLEY_WHITE_INTERFACIAL["F"], -1.13)
        self.assertEqual(WIMLEY_WHITE_INTERFACIAL["Y"], -0.94)

        # Charged Lys and Arg have positive penalties
        self.assertEqual(WIMLEY_WHITE_INTERFACIAL["K"], 0.99)
        self.assertEqual(WIMLEY_WHITE_INTERFACIAL["R"], 0.81)

        # Octanol non-polar insertion scales
        self.assertEqual(WIMLEY_WHITE_OCTANOL["W"], -2.09)
        self.assertEqual(WIMLEY_WHITE_OCTANOL["F"], -1.71)
        self.assertEqual(WIMLEY_WHITE_OCTANOL["E"], 3.63)

    def test_sequence_cleaning(self):
        """Verifies sequence standardization and non-canonical filtering."""
        raw = "  kwk-l_fkk 123 \n"
        cleaned = clean_amp_sequence(raw)
        self.assertEqual(cleaned, "KWKLFKK")

    def test_interfacial_free_energy_calculation(self):
        """Verifies additive interfacial partitioning summation."""
        # "WWW" has 3 aromatic anchors: 3 * -1.85 = -5.55 kcal/mol
        res = calculate_wimley_white_interfacial_dg("WWW")
        self.assertAlmostEqual(res["total_dg_if"], -5.55, places=2)
        self.assertAlmostEqual(res["mean_dg_if"], -1.85, places=2)
        self.assertEqual(res["aromatic_anchor_count"], 3)

        # Empty sequence edge case
        empty = calculate_wimley_white_interfacial_dg("")
        self.assertEqual(empty["total_dg_if"], 0.0)
        self.assertEqual(empty["aromatic_anchor_count"], 0)

    def test_henderson_hasselbalch_charge(self):
        """Verifies pH 7.4 net charge determination."""
        # Poly-lysine "KKKK" at pH 7.4:
        # N-term (+0.99) + 4 * Lys (+0.999) - C-term (-1.0) = ~ +4.0
        z_poly_k = calculate_net_charge_at_ph("KKKK", ph=7.4)
        self.assertTrue(3.8 < z_poly_k < 4.2)

        # Poly-glutamate "EEEE" at pH 7.4:
        # N-term (+0.99) - 4 * Glu (-0.999) - C-term (-1.0) = ~ -4.0
        z_poly_e = calculate_net_charge_at_ph("EEEE", ph=7.4)
        self.assertTrue(-4.2 < z_poly_e < -3.8)

        # Alanine tetramer "AAAA": N-term (+0.99) - C-term (-1.0) = ~ 0.0
        z_poly_a = calculate_net_charge_at_ph("AAAA", ph=7.4)
        self.assertAlmostEqual(z_poly_a, 0.0, places=1)

    def test_amphipathic_moment(self):
        """Verifies Eisenberg hydrophobic moment calculation across periodicity angles."""
        seq = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"  # Magainin derivative
        mu_alpha = calculate_amphipathic_moment(seq, angle_deg=100.0)
        mu_beta = calculate_amphipathic_moment(seq, angle_deg=160.0)

        self.assertGreaterEqual(mu_alpha, 0.0)
        self.assertLessEqual(mu_alpha, 1.0)
        self.assertGreaterEqual(mu_beta, 0.0)
        self.assertLessEqual(mu_beta, 1.0)

    def test_evaluate_membrane_activity_melittin(self):
        """Verifies biophysical evaluation of the classical hemolytic AMP Melittin."""
        melittin = "GIGAVLKVLTTGLPALISWIKRKRQQ"
        eval_res = evaluate_membrane_activity(melittin)

        self.assertEqual(eval_res["length"], 26)
        self.assertGreater(eval_res["net_charge_z"], 4.5)  # Highly cationic C-terminal tail
        self.assertGreater(eval_res["membrane_permeabilization_index"], 25.0)
        self.assertEqual(eval_res["aromatic_anchors_count"], 1)  # Trp19
        self.assertIn("Moderate", eval_res["hemolysis_tier"])

    def test_evaluate_membrane_activity_pexiganan(self):
        """Verifies high amphipathic alpha-helical pore former Pexiganan (MSI-78)."""
        pexiganan = "GIGKFLKKAKKFGKAFVKILKK"
        eval_res = evaluate_membrane_activity(pexiganan)

        self.assertEqual(eval_res["length"], 22)
        self.assertGreater(eval_res["net_charge_z"], 8.0)
        self.assertGreater(eval_res["mu_h_alpha_100deg"], 0.40)
        self.assertGreater(eval_res["membrane_permeabilization_index"], 60.0)
        self.assertEqual(eval_res["favored_conformation"], "alpha_helix")

    def test_short_sequence_error_handling(self):
        """Verifies that sequences shorter than 4 residues raise ValueError."""
        with self.assertRaises(ValueError):
            evaluate_membrane_activity("KWK")


class TestESMEmbedder(unittest.TestCase):
    """Verifies ESM-2 model inference, batching, and calibrated sequence predictions."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = ESM2Embedder(model_name="facebook/esm2_t12_35M_UR50D", device="cpu")

    def test_embedding_dimensions(self):
        """Verifies 480-dimensional mean pooling from 35M ESM-2."""
        seq = "KWKLFKKIEKVGQNIRDGIIK"
        emb = self.embedder.embed_sequence(seq)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (480,))
        self.assertFalse(np.isnan(emb).any())

    def test_sequence_embedding_statistics(self):
        """Verifies neural representation statistics from ESM-2."""
        seq = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
        res = self.embedder.analyze_sequence_embedding(seq)
        self.assertIn("embedding", res)
        self.assertEqual(res["embedding_dim"], 480)
        self.assertGreater(res["embedding_norm"], 0.0)
        self.assertIsInstance(res["embedding"], np.ndarray)

    def test_batch_embedding_consistency(self):
        """Verifies that batch embeddings match single sequence embeddings."""
        seqs = ["ALAKALAK", "KKLLKKLL", "WWFFWWFF"]
        batch_embs = self.embedder.batch_embed_sequences(seqs)
        self.assertEqual(len(batch_embs), 3)

        for seq, b_emb in zip(seqs, batch_embs):
            s_emb = self.embedder.embed_sequence(seq)
            np.testing.assert_allclose(b_emb, s_emb, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
