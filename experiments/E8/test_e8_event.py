"""Focused checks of event inference, alignment, pairing and causal ordering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from e8_event import (EventSetting, align_event_indices, common_observations,
                      compute_observations, continuous_slope, prior_event_mask,
                      tercile_labels, welch_difference)


class EventChecks(unittest.TestCase):
    def test_welch_matches_scipy(self):
        a = np.array([0.01, 0.02, -0.01, 0.04, 0.01])
        b = np.array([-0.01, -0.02, 0.01, -0.03, 0.0, -0.01])
        actual = welch_difference(a, b)
        expected = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(actual["estimate"], a.mean() - b.mean())
        self.assertAlmostEqual(actual["p_value"], expected.pvalue)
        self.assertLess(actual["ci_low"], actual["estimate"])
        self.assertGreater(actual["ci_high"], actual["estimate"])

    def test_empty_groups_do_not_become_zero(self):
        result = welch_difference(np.array([]), np.array([1.0, 2.0]))
        self.assertTrue(np.isnan(result["estimate"]))
        self.assertTrue(np.isnan(result["p_value"]))

    def test_weekend_alignment_and_outside_calendar(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        pubs = pd.Series(pd.to_datetime(["2019-12-01", "2020-01-04", "2020-01-07", "2020-02-01"]))
        result = align_event_indices(dates, pubs)
        np.testing.assert_array_equal(result, [-1, 3, 4, -1])

    def test_purge_removes_full_event_window(self):
        dates = pd.bdate_range("2020-01-01", periods=80)
        pubs = pd.Series([dates[20], dates[55]])
        result = prior_event_mask(dates, pubs)
        self.assertTrue(result[20:31].all())
        self.assertFalse(result[19])
        self.assertFalse(result[31])

    def test_rank_terciles_reproduce_e3_cut_rule(self):
        frame = pd.DataFrame({"issue": [str(i) for i in range(10)], "score": range(10)})
        groups = tercile_labels(frame, "score")
        self.assertEqual(groups.tolist(), ["low"] * 4 + ["mid"] * 3 + ["high"] * 3)

    def test_pairing_uses_issue_and_industry(self):
        left = pd.DataFrame({"issue": ["a", "a", "b"], "code": ["x", "y", "x"], "car": [1, 2, 3]})
        right = pd.DataFrame({"issue": ["a", "b", "c"], "code": ["x", "y", "x"], "car": [4, 5, 6]})
        l, r = common_observations(left, right)
        self.assertEqual(len(l), 1)
        self.assertEqual(l[["issue", "code"]].values.tolist(), r[["issue", "code"]].values.tolist())
        self.assertEqual(l.iloc[0]["car"], 1)
        self.assertEqual(r.iloc[0]["car"], 4)

    def test_hc3_slope_uses_event_sample(self):
        x = np.arange(20, dtype=float)
        noise = np.sin(x) * 0.01
        y = 0.002 * x + noise
        result = continuous_slope(x, y)
        self.assertGreater(result["estimate"], 0)
        self.assertLess(result["p_value"], 0.01)

    def test_estimation_precedes_event_and_purge_is_effective(self):
        dates = pd.bdate_range("2020-01-01", periods=150)
        events = pd.DataFrame({"issue": ["a", "b"], "publication_date": [dates[25], dates[130]]})
        market = np.sin(np.arange(150)) / 100
        industries = pd.DataFrame({"date": dates, "code": "x", "name": "X", "daily_return": market * 1.3 + 0.001})
        benchmark = {"sh000300": pd.DataFrame({"date": dates, "daily_return": market})}
        settings = [EventSetting("main", estimation_window=120), EventSetting("purged", estimation_window=120, purge=True)]
        obs, failures = compute_observations(events, industries, benchmark, settings)
        self.assertTrue((obs["estimation_end"] < obs["event_trading_date"]).all())
        purged = obs[obs["purge"]]
        self.assertEqual(len(purged), 1)
        self.assertFalse(purged["any_mpr_window_in_estimation"].any())
        self.assertEqual(purged.iloc[0]["n_purged_days"], 11)
        self.assertTrue((failures["reason"] == "insufficient_benchmark_estimation_history").any())


if __name__ == "__main__":
    unittest.main()
