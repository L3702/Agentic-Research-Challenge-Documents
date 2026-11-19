"""Focused timing, lagging, pairing and legacy reproduction checks for E8."""

import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import e8_prediction as p


class PredictionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = p.load_panel()

    def test_legacy_target_is_not_postpublication(self):
        self.assertTrue((self.panel.label_start <= self.panel.publication_date).all())

    def test_strict_labels_and_train_are_available(self):
        strict = p.strict_target(self.panel)
        self.assertTrue((strict.publication_date < strict.label_start).all())
        summary, _, pred, _ = p.evaluate(strict, "timing_test", True)
        self.assertGreater(summary["n_oos_periods"], 10)
        self.assertTrue((pred.max_train_label_end < pred.publication_date).all())

    def test_lag2_is_exact_quarter_shift(self):
        lagged = p.true_quarter_lag(self.panel)
        valid = lagged.dropna(subset=["signal", "source_issue"])
        source = self.panel.set_index(p.KEY)
        for r in valid.iloc[::101].itertuples():
            prior = str(pd.Period(r.issue, freq="Q") - 1)
            self.assertEqual(r.source_issue, prior)
            self.assertAlmostEqual(r.signal, source.loc[(prior, r.code), "signal"])
            self.assertAlmostEqual(r.mom1_z, source.loc[(prior, r.code), "mom1_z"])

    def test_main_coefficient_reproduces_e4(self):
        co, _ = p.fit_cross_sections(self.panel, "signal", "reproduce", "sentiment")
        expected = pd.read_csv(p.ROOT / "experiments/E4/e4_fmb_results.csv")
        expected = expected.loc[(expected.dictionary == "ntusd_simplified") & (expected.coefficient == p.PRIMARY)].iloc[0]
        self.assertEqual(len(co), int(expected.n_periods))
        self.assertAlmostEqual(co.signal.mean(), expected["mean"], places=10)

    def test_common_sample_and_original_oos_outcomes(self):
        win = p.winsor_past(self.panel)
        np.testing.assert_allclose(win.actual_evaluation, self.panel.actual_evaluation)
        altered = self.panel.copy()
        altered.loc[altered.index[::11], "signal"] = np.nan
        result, co, pred, _ = p.evaluate(altered, "paired_test", False, self.panel)
        obs = co.groupby(["target_quarter", "model_role"])["n_obs"].first().unstack()
        self.assertTrue((obs.max(axis=1) == obs.min(axis=1)).all())
        counts = pred.groupby(["target_quarter", "model_role"]).size().unstack()
        self.assertTrue((counts.max(axis=1) == counts.min(axis=1)).all())
        self.assertTrue(np.isfinite(result["baseline_ic"]))

    def test_terciles_do_not_use_future(self):
        old = p.historical_terciles(self.panel)
        cut = self.panel.publication_date.quantile(.7)
        changed = self.panel.copy()
        changed.loc[changed.publication_date > cut, "sentiment_ntusd_simplified"] += 1000
        new = p.historical_terciles(changed)
        early = old.publication_date <= cut
        np.testing.assert_allclose(old.loc[early, "signal"], new.loc[early, "signal"], equal_nan=True)

    def test_tercile_three_roles_use_identical_training_quarters(self):
        ordinal = p.historical_terciles(self.panel)
        result, co, pred, _ = p.evaluate(ordinal, "ordinal_common_test", False, self.panel)
        counts = co.groupby("model_role")["target_quarter"].nunique()
        self.assertEqual(counts.nunique(), 1)
        signatures = pred.groupby("target_quarter")["train_quarter_set_sha256"].nunique()
        self.assertTrue((signatures == 1).all())
        self.assertGreater(result["raw_fit_periods_notext"], result["common_identifiable_periods"])

    def test_missing_forecast_never_falls_back_to_unpaired_metrics(self):
        early = self.panel.loc[self.panel.publication_date.dt.year < 2008]
        result, _, pred, _ = p.evaluate(early, "no_oos_test", False, self.panel)
        self.assertTrue(pred.empty)
        self.assertTrue(np.isnan(result["incremental_oos_r2"]))
        self.assertTrue(np.isnan(result["baseline_ic"]))
        self.assertEqual(result["n_oos_periods"], 0)
        self.assertEqual(result["oos_pairing_status"], "unavailable_one_or_more_roles_have_no_forecasts")


if __name__ == "__main__":
    unittest.main()
