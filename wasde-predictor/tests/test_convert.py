"""Tests for the real-data converters (pure functions, fixture inputs)."""
import unittest

from wasde_predictor import convert


class NassConditionTests(unittest.TestCase):
    def test_sums_good_and_excellent_for_national(self):
        records = [
            {"agg_level_desc": "NATIONAL", "week_ending": "2024-08-11", "unit_desc": "PCT GOOD", "Value": "50"},
            {"agg_level_desc": "NATIONAL", "week_ending": "2024-08-11", "unit_desc": "PCT EXCELLENT", "Value": "15"},
            {"agg_level_desc": "STATE", "state_alpha": "IA", "week_ending": "2024-08-11", "unit_desc": "PCT GOOD", "Value": "99"},
        ]
        rows = convert.nass_condition_to_rows(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "65")          # 50 + 15, state total only
        self.assertEqual(rows[0]["state"], "US TOTAL")

    def test_skips_weeks_missing_a_category(self):
        records = [{"agg_level_desc": "NATIONAL", "week_ending": "2024-08-11",
                    "unit_desc": "PCT GOOD", "Value": "50"}]
        self.assertEqual(convert.nass_condition_to_rows(records), [])

    def test_handles_suppressed_values(self):
        records = [
            {"agg_level_desc": "NATIONAL", "week_ending": "2024-08-11", "unit_desc": "PCT GOOD", "Value": "(D)"},
            {"agg_level_desc": "NATIONAL", "week_ending": "2024-08-11", "unit_desc": "PCT EXCELLENT", "Value": "15"},
        ]
        self.assertEqual(convert.nass_condition_to_rows(records), [])  # good was suppressed


class DroughtTests(unittest.TestCase):
    def test_cumulative_uses_d2_column(self):
        records = [{"ValidEnd": "2024-08-13", "D2": "30", "D3": "10", "D4": "2"}]
        rows = convert.drought_to_rows(records, region="US CORN BELT", cumulative=True)
        self.assertEqual(rows[0]["value"], "30")
        self.assertEqual(rows[0]["week_ending"], "2024-08-13")
        self.assertEqual(rows[0]["region"], "US CORN BELT")

    def test_noncumulative_sums_categories(self):
        records = [{"MapDate": "20240813", "D2": "30", "D3": "10", "D4": "2"}]
        rows = convert.drought_to_rows(records, cumulative=False)
        self.assertEqual(rows[0]["value"], "42")            # 30+10+2
        self.assertEqual(rows[0]["week_ending"], "2024-08-13")  # YYYYMMDD normalized


class PaceSurpriseTests(unittest.TestCase):
    def test_surprise_is_deviation_from_week_of_year_mean(self):
        # same ISO week across two years; values 10 and 20 -> mean 15 -> surprises -5, +5
        records = [
            {"week_ending": "2023-08-14", "v": "10"},
            {"week_ending": "2024-08-12", "v": "20"},
        ]
        rows = convert.pace_surprise_rows(records, date_col="week_ending", value_col="v",
                                          key_column="commodity", key="CORN", metric="EXPORT_PACE_SURPRISE")
        by_date = {r["week_ending"]: float(r["value"]) for r in rows}
        self.assertAlmostEqual(by_date["2023-08-14"], -5.0)
        self.assertAlmostEqual(by_date["2024-08-12"], 5.0)
        self.assertEqual(rows[0]["commodity"], "CORN")


if __name__ == "__main__":
    unittest.main()
