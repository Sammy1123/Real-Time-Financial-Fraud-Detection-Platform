"""
Unit tests for pipeline logic — no server, no Kafka, no Kaggle data needed.

Covers:
  - chain_edges           (GNN graph edge builder)
  - compute_psi           (drift detection)
  - validate_message      (Kafka producer schema)
  - preprocess_transaction (shared inference preprocessing)
Run with:  pytest tests/test_pipeline_logic.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── GNN chain_edges ────────────────────────────────────────────────────────
class TestChainEdges:
    def test_basic_group_linking(self):
        from models.train_gnn import chain_edges
        # three nodes share code 7 -> bidirectional chain with offsets 1 and 2
        src, dst = chain_edges(np.array([7, 7, 7]))
        links = set(zip(src.tolist(), dst.tolist()))
        assert {(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)} <= links

    def test_distinct_codes_not_linked(self):
        from models.train_gnn import chain_edges
        src, dst = chain_edges(np.array([1, 2, 3]))
        assert len(src) == 0

    def test_nan_code_skipped(self):
        from models.train_gnn import chain_edges
        src, dst = chain_edges(np.array([-1, -1, -1]))
        assert len(src) == 0

    def test_bidirectional(self):
        from models.train_gnn import chain_edges
        src, dst = chain_edges(np.array([5, 5]))
        assert src.tolist() == [0, 1]
        assert dst.tolist() == [1, 0]

    def test_max_links_bounds(self):
        from models.train_gnn import chain_edges
        # 6 nodes same code, max_links=1 -> exactly 5 unique forward links
        src, dst = chain_edges(np.array([3] * 6), max_links=1)
        assert len(src) == 5 * 2  # bidirectional

    def test_output_dtype_int64(self):
        from models.train_gnn import chain_edges
        src, dst = chain_edges(np.array([9, 9]))
        assert src.dtype == np.int64 and dst.dtype == np.int64


# ── compute_psi ────────────────────────────────────────────────────────────
class TestComputePSI:
    def _run(self, ref, cur):
        # import from monitoring script without executing its main flow
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "drift", ROOT / "monitoring" / "drift_detection.py")
        mod = importlib.util.module_from_spec(spec)
        # load only the function definitions by exec'ing the source segment
        src = Path(ROOT / "monitoring" / "drift_detection.py").read_text()
        func_src = src[src.index("def compute_psi"):src.index("print(\"Computing PSI")]
        namespace = {"np": np}
        exec(func_src, namespace)
        return namespace["compute_psi"](np.asarray(ref), np.asarray(cur))

    def test_identical_distributions_zero_psi(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1, 10_000)
        assert self._run(vals, vals) < 1e-6

    def test_shifted_distribution_high_psi(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 10_000)
        cur = rng.normal(4, 1, 10_000)  # massive shift
        psi = self._run(ref, cur)
        assert psi > 1.0

    def test_moderate_shift_moderate_psi(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 10_000)
        cur = rng.normal(0.3, 1, 10_000)
        psi = self._run(ref, cur)
        assert 0.0 < psi < 1.0

    def test_constant_input_returns_zero(self):
        # a single unique breakpoint => degenerate bins => 0.0
        assert self._run([1.0] * 100, [1.0] * 100) == 0.0

    def test_symmetric_measure_approx(self):
        """PSI with swapped arguments lands in the same magnitude band —
        exact symmetry doesn't hold because bins derive from the reference."""
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, 5000), rng.normal(0.5, 1, 5000)
        psi_ab, psi_ba = self._run(a, b), self._run(b, a)
        assert psi_ab > 0 and psi_ba > 0
        assert abs(psi_ab - psi_ba) / max(psi_ab, psi_ba) < 0.5


# ── validate_message ───────────────────────────────────────────────────────
from streaming.kafka_producer import validate_message


class TestValidateMessage:
    def _valid(self, **overrides):
        msg = {"TransactionID": 1, "TransactionDT": 1000, "TransactionAmt": 50.0}
        msg.update(overrides)
        return msg

    def test_valid_message_passes(self):
        ok, reason = validate_message(self._valid(card1=9500.0))
        assert ok, reason

    def test_missing_required_field(self):
        ok, reason = validate_message({"TransactionDT": 1000})
        assert not ok and "TransactionID" in reason

    def test_missing_amt(self):
        ok, reason = validate_message({"TransactionID": 1, "TransactionDT": 1})
        assert not ok and "TransactionAmt" in reason

    def test_wrong_type_rejected(self):
        ok, reason = validate_message(self._valid(TransactionID="abc"))
        assert not ok and "type" in reason

    def test_negative_amount_rejected(self):
        ok, reason = validate_message(self._valid(TransactionAmt=-1))
        assert not ok and "positive" in reason

    def test_zero_amount_rejected(self):
        ok, _ = validate_message(self._valid(TransactionAmt=0))
        assert not ok

    def test_numeric_field_with_string_rejected(self):
        ok, reason = validate_message(self._valid(card1="visa"))
        assert not ok and "card1" in reason

    def test_none_numeric_field_allowed(self):
        ok, _ = validate_message(self._valid(card1=None, addr1=None))
        assert ok

    def test_negative_id_rejected(self):
        ok, reason = validate_message(self._valid(TransactionID=-5))
        assert not ok

    def test_non_dict_rejected(self):
        ok, reason = validate_message([1, 2, 3])
        assert not ok and "JSON object" in reason


# ── preprocess_transaction ────────────────────────────────────────────────
from api.inference import preprocess_transaction, FEATURES


class TestPreprocessTransaction:
    def _base_tx(self, **overrides):
        tx = {"TransactionID": 42, "TransactionDT": 7200,
              "TransactionAmt": 120.0, "card1": 9500.0,
              "card4": "visa", "ProductCD": "W"}
        tx.update(overrides)
        return tx

    def test_returns_correct_length(self):
        vec = preprocess_transaction(self._base_tx())
        assert len(vec) == len(FEATURES)

    def test_log_amt_engineered(self):
        vec = preprocess_transaction(self._base_tx())
        idx = FEATURES.index("log_amt")
        assert vec[idx] == pytest.approx(np.log1p(120.0), rel=1e-5)

    def test_hour_derived_from_dt(self):
        vec = preprocess_transaction(self._base_tx())   # 7200s = 02:00
        idx = FEATURES.index("hour")
        assert vec[idx] == 2.0

    def test_missing_fields_filled_with_sentinel(self):
        bare = preprocess_transaction({"TransactionAmt": 10.0, "TransactionDT": 0})
        assert -999.0 in bare  # unmapped features use the sentinel

    def test_no_nan_in_output(self):
        vec = preprocess_transaction(self._base_tx())
        assert not np.isnan(vec).any()

    def test_non_numeric_categorical_does_not_crash(self):
        vec = preprocess_transaction(self._base_tx(card6="debit"))
        assert len(vec) == len(FEATURES)

    def test_negative_amount_handled(self):
        vec = preprocess_transaction(self._base_tx(TransactionAmt=-5.0))
        idx = FEATURES.index("log_amt")
        assert vec[idx] == pytest.approx(np.log1p(0.0))  # clamped, no NaN/-inf

    def test_output_is_float32(self):
        assert preprocess_transaction(self._base_tx()).dtype == np.float32
