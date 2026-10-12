import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import state, summarize, birth_for_label, frozen_subtypes, binary_summary, decisions


class StateTests(unittest.TestCase):
    def test_states(self):
        self.assertEqual(state(160, 161, 1), "W")
        self.assertEqual(state(160, 160, 1), "C")
        self.assertEqual(state(160, 160, 160), "K")

    def test_inconsistent_all_correct_rejected(self):
        with self.assertRaises(ValueError):
            state(160, 161, 160)

    def test_t10_t9_class_range(self):
        partition = [20] * 10
        self.assertEqual(birth_for_label(160, partition), 9)
        self.assertEqual(birth_for_label(179, partition), 9)
        self.assertEqual(birth_for_label(180, partition), 10)
        with self.assertRaises(ValueError):
            birth_for_label(200, partition)

    def test_all_nine_transitions_and_identities(self):
        pairs = [{"sample_id": f"{a}{b}", "y": 0, "reference_state": a, "target_state": b}
                 for a in "WCK" for b in "WCK"]
        row = summarize(pairs, "IA10", 9, "T9_all", "G-S")
        self.assertEqual(row["N"], 9)
        self.assertEqual((row["local_repair"], row["local_harm"], row["local_net"]), (2, 2, 0))
        self.assertEqual((row["all_repair"], row["all_harm"], row["all_net"]), (2, 2, 0))
        self.assertAlmostEqual(row["reference_local_accuracy_pct"], 200/3)
        self.assertAlmostEqual(row["reference_all_accuracy_pct"], 100/3)

    def test_wc_hidden_repair_and_kw_harm(self):
        pairs = [{"sample_id": "a", "y": 0, "reference_state": "W", "target_state": "C"},
                 {"sample_id": "b", "y": 1, "reference_state": "K", "target_state": "W"}]
        row = summarize(pairs, "IA10", 10, "T9_all", "G-S")
        self.assertEqual(row["local_net"], 0)
        self.assertEqual(row["all_net"], -1)
        self.assertEqual(row["all_delta_pp"], -50)

    def test_macro_is_not_micro(self):
        pairs = [{"sample_id": str(i), "y": 0 if i < 3 else 1, "reference_state": "K" if i < 3 else "W", "target_state": "K"}
                 for i in range(4)]
        row = summarize(pairs, "IA10", 9, "T9_all", "G-S")
        self.assertEqual(row["reference_all_accuracy_pct"], 75)
        self.assertEqual(row["reference_macro_all_accuracy_pct"], 50)

    def test_empty_cohort(self):
        row = summarize([], "IA10", 9, "T9_A_within", "G-S")
        self.assertEqual(row["N"], 0)
        self.assertIsNone(row["local_delta_pp"])
        self.assertIsNone(row["target_macro_all_accuracy_pct"])

    def test_fixed_subtypes_repeats_not_new_branch(self):
        trajectory = {"a": {"sample_id": "a", "error_type": "A_persistent_wrong", "birth_task": "9",
                              "true_label": "160", "pred_Tbirth": "1"}}
        row = {"sample_id": "a", "group": "A", "y": "160", "task": "9", "local_pred": "160", "all_pred": "1", "A_location": "A-cross"}
        self.assertEqual(frozen_subtypes(trajectory, [row, row.copy()]), {"a": "A-cross"})
        with self.assertRaises(ValueError):
            frozen_subtypes(trajectory, [row, {**row, "local_pred": "161", "A_location": "A-within"}])

    def test_collateral_binary_net(self):
        rows = [{"reference_correct": False, "target_correct": True},
                {"reference_correct": True, "target_correct": False},
                {"reference_correct": True, "target_correct": False}]
        row = binary_summary(rows, "CAR10", 10, "historical", "G-S")
        self.assertEqual((row["repair"], row["harm"], row["net"]), (1, 2, -1))

    def test_common_signal_does_not_follow_one_dataset(self):
        rows = []
        for protocol, net in (("IA10", 1), ("CAR10", -5)):
            for task in (9, 10):
                rows.append({"protocol": protocol, "time": f"T{task}", "comparison": "G-S", "cohort": "T9_all",
                             "local_net": net if task == 9 else 0, "local_delta_pp": net if task == 9 else 0,
                             "all_net": 0})
        table, decision = decisions(rows)
        self.assertEqual(decision["IA10_LOCAL_SIGNAL"], "YES")
        self.assertEqual(decision["CAR10_LOCAL_SIGNAL"], "NO")
        self.assertEqual(decision["LOCAL_SIGNAL_PERSISTS_TO_T10"], "NO")
        self.assertEqual(table[0]["SPEC_NONNEGATIVE_T10_CONDITION"], "YES")
        self.assertEqual(decision["GENERATION_DIRECTION"], "STOP")


if __name__ == "__main__":
    unittest.main()
