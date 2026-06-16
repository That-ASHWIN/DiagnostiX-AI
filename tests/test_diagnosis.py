import unittest

from diagnosis import create_feature_row, get_artifact, predict_fault


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = get_artifact()

    def test_feature_row_preserves_user_inputs(self):
        row = create_feature_row(
            device="Mobile",
            age_months=32,
            daily_usage_hours=2,
            failure_after_months=29,
            usage_type="Developer",
            symptom1="Sample Symptom A",
            symptom2="Sample Symptom B",
            symptom3="Sample Symptom C",
        )

        self.assertEqual(row["Device"], "Mobile")
        self.assertEqual(row["Age_Months"], 32)
        self.assertEqual(row["Symptom1"], "Sample Symptom A")
        self.assertEqual(row["Symptom3"], "Sample Symptom C")

    def test_input_options_structure(self):
        options = self.artifact["input_options"]
        self.assertTrue(options["devices"])
        self.assertTrue(options["usage_types"])
        for device in options["devices"]:
            combinations = options["symptom_combinations_by_device"][device]
            self.assertTrue(combinations)
            self.assertIn("Symptom1", combinations[0])
            self.assertIn("Symptom2", combinations[0])
            self.assertIn("Symptom3", combinations[0])

    def test_prediction_is_valid(self):
        options = self.artifact["input_options"]
        device = options["devices"][0]
        combo = options["symptom_combinations_by_device"][device][0]
        row = create_feature_row(
            device=device,
            age_months=24,
            daily_usage_hours=8,
            failure_after_months=18,
            usage_type=options["usage_types"][0],
            symptom1=combo["Symptom1"],
            symptom2=combo["Symptom2"],
            symptom3=combo["Symptom3"],
        )
        result = predict_fault(self.artifact, row)

        self.assertIn("fault", result)
        self.assertEqual(len(result["alternatives"]), 3)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn(
            result["fault"], list(self.artifact["model"].classes_)
        )


if __name__ == "__main__":
    unittest.main()
