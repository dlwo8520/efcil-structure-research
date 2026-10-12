import unittest

from scripts.aggregate_results import EXPERIMENTS, common_comparable_seeds, mean_std_rows


class AggregateResultsTests(unittest.TestCase):
    def test_only_common_complete_matching_protocol_seed_is_used(self):
        runs = {
            experiment: {
                7: {"complete": True, "protocol_signature": "same"},
                9: {"complete": experiment != "E3", "protocol_signature": "same"},
            }
            for experiment in EXPERIMENTS
        }
        exclusions = []
        self.assertEqual(common_comparable_seeds(runs, exclusions), [7])

    def test_mean_rows_do_not_fill_missing_values_with_zero(self):
        rows = mean_std_rows([], [])
        self.assertEqual(len(rows), 8)
        self.assertIsNone(rows[0]["Inc-Acc_mean"])
        self.assertEqual(rows[0]["valid_seed_count"], 0)


if __name__ == "__main__":
    unittest.main()
