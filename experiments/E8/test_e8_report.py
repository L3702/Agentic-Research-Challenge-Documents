"""Aggregation checks for E8 p-values and deliverable coverage."""

import unittest

import numpy as np
import pandas as pd

from run_e8 import adjust_tests


class ReportTests(unittest.TestCase):
    def test_p_adjustments_are_separate_and_preserve_missing(self):
        frame = pd.DataFrame({
            "block": ["event", "event", "prediction", "prediction", "prediction"],
            "timing_protocol": [None, None, "legacy_e4_overlap", "strict_post_publication", "strict_post_publication"],
            "p_value": [0.01, 0.2, 0.02, 0.03, np.nan],
            "mean_car_p": [0.01, 0.8, np.nan, np.nan, np.nan],
        })
        out = adjust_tests(frame)
        np.testing.assert_allclose(out.loc[:1, "p_value_bh"], [0.02, 0.2])
        self.assertAlmostEqual(out.loc[2, "p_value_bh"], 0.02)
        self.assertAlmostEqual(out.loc[3, "p_value_bh"], 0.03)
        self.assertTrue(pd.isna(out.loc[4, "p_value_bh"]))
        self.assertFalse(bool(out.loc[4, "raw_significant_5pct"]))
        valid = out.p_value.notna()
        self.assertTrue((out.loc[valid, "p_value_holm"] >= out.loc[valid, "p_value"]).all())

    def test_invalid_p_values_do_not_silently_pass(self):
        with self.assertRaises(AssertionError):
            adjust_tests(pd.DataFrame({"block": ["event"], "p_value": [-0.2]}))


if __name__ == "__main__":
    unittest.main()
