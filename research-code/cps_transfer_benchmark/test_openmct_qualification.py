#!/usr/bin/env python3
import tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from qualify_openmct import FEATURES, bind_extracted_files_to_archive, free_run, model_fit


class QualificationTests(unittest.TestCase):
    def test_no_future_measurement_feature(self):
        self.assertNotIn("speed_kp1", FEATURES)

    def test_free_run_ignores_future_measurements(self):
        n = 40
        df = pd.DataFrame({"MEAS": np.linspace(2, 8, n), "REF": np.linspace(3, 9, n),
                           "PWM": np.linspace(10, 20, n), "DT_ms": np.full(n, 10.0)})
        x = np.column_stack([np.ones(20), np.ones(20), np.zeros(20), np.zeros(20), np.zeros(20), np.full(20,10.)])
        fit = model_fit(x, np.ones(20), np.ones(6, bool))
        _, first = free_run(df, fit)
        changed = df.copy(); changed.loc[2:, "MEAS"] += 10000
        _, second = free_run(changed, fit)
        np.testing.assert_array_equal(first, second)

    def test_archive_extract_binding_fails_on_changed_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); raw = root / "raw"; raw.mkdir()
            extracted = raw / "record.txt"; extracted.write_bytes(b"original")
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("dataset/record.txt", b"original")
            binding = bind_extracted_files_to_archive(archive, raw, [extracted])
            self.assertEqual(binding["record.txt"]["archive_member"], "dataset/record.txt")
            extracted.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from archive"):
                bind_extracted_files_to_archive(archive, raw, [extracted])


if __name__ == "__main__": unittest.main()
