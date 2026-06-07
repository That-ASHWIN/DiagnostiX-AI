import unittest

from diagnosis import create_feature_row, load_artifact, predict_fault


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_artifact()

    def test_feature_row_preserves_user_inputs(self):
        row = create_feature_row(
            device="Mobile",
            age_months=32,
            daily_usage_hours=2,
            failure_after_months=29,
            usage_type="Developer",
            symptom1="Recording Failure",
            symptom2="Low Mic Volume",
            symptom3="Calls Connected",
        )

        self.assertEqual(row["Device"], "Mobile")
        self.assertEqual(row["Symptom1"], "Recording Failure")
        self.assertEqual(row["Symptom2"], "Low Mic Volume")
        self.assertEqual(row["Symptom3"], "Calls Connected")

    def test_artifact_contains_valid_symptom_combinations(self):
        options = self.artifact["input_options"]
        mobile_combinations = options["symptom_combinations_by_device"]["Mobile"]

        self.assertIn(
            {
                "Symptom1": "Recording Failure",
                "Symptom2": "Low Mic Volume",
                "Symptom3": "Calls Connected",
            },
            mobile_combinations,
        )

    def test_distinct_symptoms_produce_distinct_predictions(self):
        cases = [
            create_feature_row(
                "Laptop",
                22,
                9,
                16,
                "Developer",
                "Overheating",
                "Fan Noise",
                "Auto Shutdown",
            ),
            create_feature_row(
                "Desktop",
                58,
                5,
                57,
                "Normal",
                "Game Crashes",
                "Visual Glitches",
                "High GPU Temp",
            ),
            create_feature_row(
                "Mobile",
                32,
                2,
                29,
                "Developer",
                "Recording Failure",
                "Low Mic Volume",
                "Calls Connected",
            ),
        ]

        predictions = [
            predict_fault(self.artifact, case)["fault"] for case in cases
        ]

        self.assertEqual(
            predictions,
            ["Cooling Fan", "GPU", "Microphone"],
        )


if __name__ == "__main__":
    unittest.main()
