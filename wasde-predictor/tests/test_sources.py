"""Tests that the condition and drought loaders read their sample CSVs."""
import unittest
from pathlib import Path

from wasde_predictor import condition, weather

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


class LoaderTests(unittest.TestCase):
    def test_condition_loads_us_total(self):
        readings = condition.load_condition(SAMPLE / "crop_condition_sample.csv")
        self.assertTrue(readings)
        self.assertTrue(any(r.key == condition.US_TOTAL for r in readings))
        self.assertTrue(all(0 <= r.value <= 100 for r in readings))
        self.assertEqual(readings, sorted(readings, key=lambda r: r.week_ending))

    def test_drought_loads_corn_belt(self):
        readings = weather.load_drought(SAMPLE / "drought_sample.csv")
        self.assertTrue(readings)
        self.assertTrue(any(r.key == weather.CORN_BELT for r in readings))
        self.assertTrue(all(0 <= r.value <= 100 for r in readings))


if __name__ == "__main__":
    unittest.main()
