import io
import csv
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from news15.multisource import (PartialZipReader, apply_claim_reservations, claim_binary,
                               complete_lines, fakecorpus_rows, parse_multifc, publisher_proxy)


class MultiSourceContracts(unittest.TestCase):
    def test_weak_publisher_labels_do_not_turn_bias_or_satire_into_fake(self):
        self.assertEqual(publisher_proxy("fake"), 1)
        self.assertEqual(publisher_proxy("reliable"), 0)
        for value in ["satire", "bias", "political", "unreliable", "conspiracy", ""]:
            self.assertIsNone(publisher_proxy(value))

    def test_mixed_and_numeric_claim_scales_are_not_guessed(self):
        self.assertEqual(claim_binary("true"), 0)
        self.assertEqual(claim_binary("false"), 1)
        for value in ["0", "1", "half-true", "mostly true", "promise kept", None]:
            self.assertIsNone(claim_binary(value))

    def test_multifc_test_has_no_label_and_report_never_becomes_claim_text(self):
        raw = ["pomt-123", "An original claim", "/factcheck/", "SECRET VERDICT FALSE", "c", "s", "checker", "tags", "False: headline", "2018", "2017", "entities"]
        row = parse_multifc(raw, "test")
        self.assertIsNone(row["label"])
        self.assertIsNone(row["original_label"])
        self.assertEqual(row["text"], "An original claim")
        self.assertEqual(row["source_url"], "/factcheck/")
        self.assertNotIn("SECRET", str(row))
        self.assertNotIn("False: headline", str(row))

    def test_malformed_multifc_metadata_is_quarantined(self):
        self.assertIsNone(parse_multifc(["x"]*10, "train"))

    def test_cross_dataset_test_copy_reserves_the_whole_claim_cluster(self):
        rows = [{"group_id": "same", "label": 1, "original_split": "train", "collection": "LIAR", "original_label": "false"},
                {"group_id": "same", "label": None, "original_split": "test", "collection": "MultiFC", "original_label": None}]
        apply_claim_reservations(rows)
        self.assertTrue(all(r["eligibility"] == "reserved_original_test" for r in rows))

    def test_conflicting_binary_claim_labels_are_not_majority_voted(self):
        rows = [{"group_id": "same", "label": label, "original_split": "train", "collection": "LIAR", "original_label": str(label)} for label in [0, 1, 1, 1]]
        summary = apply_claim_reservations(rows)
        self.assertEqual(summary["conflicting_binary_label_groups"], 1)
        self.assertTrue(all(r["label"] is None and r["eligibility"].startswith("quarantine") for r in rows))

    def test_incomplete_final_physical_line_is_discarded(self):
        self.assertEqual(list(complete_lines(io.StringIO("complete\npartial"))), ["complete\n"])

    def test_carriage_returns_and_bad_rows_do_not_truncate_the_dataset(self):
        fields = ["id", "domain", "type", "url", "content", "title", "source"]
        text = io.StringIO(newline="")
        writer = csv.writer(text, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(fields)
        writer.writerow(["one", "site", "fake", "url", "Before\rafter", "title", "origin"])
        writer.writerow(["broken"])
        writer.writerow(["two", "site", "reliable", "url", "Another whole record", "title", "origin"])
        payload = text.getvalue().encode()
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(payload)+compressor.flush()
        name = b"news_cleaned_2018_02_13.csv"
        header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 8, 0, 0, 0, len(compressed), len(payload), len(name), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"part.z01"
            path.write_bytes(b"PK\x07\x08"+header+name+compressed)
            rows = list(fakecorpus_rows(path))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][1]["content"], "Before\rafter")
        self.assertEqual(rows[1][1]["_import_error"], "invalid_csv_columns")
        self.assertEqual(rows[2][1]["id"], "two")

    def test_first_split_zip_volume_can_be_read_without_false_crc_claim(self):
        text = b"id,content\n1,a real original article\n"
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(text)+compressor.flush()
        name = b"news_cleaned_2018_02_13.csv"
        header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 8, 0, 0, 0, len(compressed), len(text), len(name), 0)
        stream = io.BytesIO(b"PK\x07\x08"+header+name+compressed)
        with io.BufferedReader(PartialZipReader(stream)) as reader:
            self.assertEqual(reader.read(), text)


if __name__ == "__main__":
    unittest.main()
