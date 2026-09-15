import unittest

from native_fullgame_action_table import (
    canonical_action_table,
    contact_mask,
    movement_mask,
    stale_label_scan_patterns,
)


class ActionTableTests(unittest.TestCase):
    def test_runtime_action_table_has_all_ten_indices(self):
        table = canonical_action_table()
        self.assertEqual(list(table), list(range(10)))
        self.assertEqual(table[0]["key"], "W")
        self.assertEqual(table[1]["key"], "A")
        self.assertEqual(table[2]["key"], "S")
        self.assertEqual(table[3]["key"], "D")
        self.assertEqual(table[4]["key"], "Space")
        self.assertEqual(table[5]["key"], "H")
        self.assertEqual(table[6]["key"], "L")
        self.assertEqual(table[7]["key"], "J")
        self.assertEqual(table[8]["key"], "K")
        self.assertEqual(table[9]["key"], "G")

    def test_high_risk_indices_use_runtime_semantics(self):
        table = canonical_action_table()
        self.assertEqual(table[5]["meaning"], "pickup_drop")
        self.assertEqual(table[6]["meaning"], "dash_dodge")
        self.assertEqual(table[7]["meaning"], "light_attack")
        self.assertEqual(table[8]["meaning"], "heavy_attack")
        self.assertEqual(table[9]["meaning"], "taunt")
        self.assertTrue(all(table[index]["trace_required"] for index in (5, 6, 7, 8, 9)))

    def test_masks_are_current_but_not_semantic_proof(self):
        self.assertEqual(movement_mask(), [0, 1, 2, 3, 4, 6])
        self.assertEqual(contact_mask(), [0, 1, 2, 3, 4, 6, 7])

    def test_table_and_masks_are_defensive_copies(self):
        table = canonical_action_table()
        table[5]["meaning"] = "stale"
        movement = movement_mask()
        contact = contact_mask()
        movement.append(5)
        contact.append(8)

        self.assertEqual(canonical_action_table()[5]["meaning"], "pickup_drop")
        self.assertEqual(movement_mask(), [0, 1, 2, 3, 4, 6])
        self.assertEqual(contact_mask(), [0, 1, 2, 3, 4, 6, 7])

    def test_stale_label_patterns_are_explicit(self):
        patterns = stale_label_scan_patterns()
        self.assertIn("5=dodge", patterns)
        self.assertIn("6=pickup", patterns)
        self.assertIn("9=throw", patterns)
        self.assertIn("indices_5_8_9_grouped_as_tools", patterns)


if __name__ == "__main__":
    unittest.main()
