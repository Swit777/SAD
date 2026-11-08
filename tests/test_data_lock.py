from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from news15.data_run_lock import exclusive_data_run
from news15.multisource import BatchedWriter


class DataWriterIsolationTests(unittest.TestCase):
    def test_second_import_is_rejected_while_first_owns_the_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"import.lock"
            with exclusive_data_run(path):
                with self.assertRaises(RuntimeError):
                    with exclusive_data_run(path):
                        self.fail("Concurrent imports must not share an output release.")
            with exclusive_data_run(path):
                pass

    def test_failed_import_releases_its_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"import.lock"
            with self.assertRaises(ValueError):
                with exclusive_data_run(path):
                    raise ValueError("A source record failed validation.")
            with exclusive_data_run(path):
                pass

    def test_open_writers_use_different_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"records.parquet"
            schema = pa.schema([("value",pa.string())])
            first = BatchedWriter(path,schema)
            second = BatchedWriter(path,schema)
            self.assertNotEqual(first.temporary,second.temporary)
            first.add({"value":"first"})
            second.add({"value":"second"})
            first.close()
            self.assertEqual(pq.read_table(path)["value"].to_pylist(),["first"])
            second.close()
            self.assertEqual(pq.read_table(path)["value"].to_pylist(),["second"])


if __name__ == "__main__":
    unittest.main()
