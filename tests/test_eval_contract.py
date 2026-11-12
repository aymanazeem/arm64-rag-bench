import hashlib
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/corpus.json"
DATASET = ROOT / "eval/development.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class EvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.dataset = json.loads(DATASET.read_text())
        cls.sources = {item["id"]: item for item in cls.manifest["sources"]}
        cls.corpora = {item["id"]: item for item in cls.manifest["corpora"]}

    def test_manifest_and_case_contract(self):
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["canonical_format"], "utf8-lf-v1")
        self.assertEqual(self.dataset["schema_version"], 1)
        self.assertEqual(self.dataset["split"], "development")
        self.assertEqual(self.dataset["manifest_sha256"], sha256(MANIFEST.read_bytes()))
        self.assertEqual(ROOT / self.dataset["manifest_path"], MANIFEST)
        self.assertFalse(self.dataset["provenance"]["human_validated"])
        self.assertEqual(len(self.sources), len(self.manifest["sources"]))
        self.assertEqual(len(self.corpora), len(self.manifest["corpora"]))
        for source in self.sources.values():
            with self.subTest(source=source["id"]):
                path = Path(source["local_path"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertIn(source["kind"], {"upstream", "fixture"})
                self.assertIn(source["license_id"], self.manifest["licenses"])
                self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(source["canonical_sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(source["bytes"], 0)
                self.assertGreater(source["canonical_chars"], 0)
                if source["kind"] == "upstream":
                    self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
                    self.assertIn(f"/{source['revision']}/", source["url"])
                    self.assertTrue(source["url"].startswith("https://"))
        for corpus in self.corpora.values():
            self.assertTrue(corpus["source_ids"])
            self.assertEqual(len(corpus["source_ids"]), len(set(corpus["source_ids"])))
            self.assertLessEqual(set(corpus["source_ids"]), self.sources.keys())
        evidence = self.dataset["evidence"]
        for item in evidence.values():
            source = self.sources[item["source_id"]]
            self.assertGreaterEqual(item["start"], 0)
            self.assertGreater(item["end"], item["start"])
            self.assertLessEqual(item["end"], source["canonical_chars"])
            self.assertGreaterEqual(item["lines"][0], 1)
            self.assertGreaterEqual(item["lines"][1], item["lines"][0])
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
        cases = self.dataset["cases"]
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        tags = set()
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertRegex(case["id"], r"^dev-\d{3}$")
                self.assertTrue(case["family"])
                self.assertTrue(case["question"].strip())
                self.assertIn(case["origin"], {"source_derived", "synthetic"})
                self.assertIn(
                    case["answerability"], {"answerable", "unanswerable", "conflicting"}
                )
                self.assertTrue(case["required_facts"])
                self.assertTrue(case["unacceptable_claims"])
                self.assertTrue(all(fact.strip() for fact in case["required_facts"]))
                scope = set(self.corpora[case["corpus_id"]]["source_ids"])
                for key in case["evidence_ids"]:
                    self.assertIn(evidence[key]["source_id"], scope)
                if case["answerability"] == "unanswerable":
                    self.assertFalse(case["evidence_ids"])
                    self.assertTrue(case["absence_rationale"])
                else:
                    self.assertTrue(case["evidence_ids"])
                if case["answerability"] == "conflicting":
                    self.assertGreaterEqual(len(case["evidence_ids"]), 2)
                tags.update(case["tags"])
        self.assertLessEqual(
            {
                "literal_flag",
                "paraphrase",
                "multi_source",
                "unanswerable",
                "conflict",
                "instruction_in_document",
            },
            tags,
        )

    def check_source_bytes_and_spans(self, source):
        path = ROOT / source["local_path"]
        self.assertTrue(path.is_file(), f"Missing source: {source['local_path']}")
        raw = path.read_bytes()
        self.assertEqual(len(raw), source["bytes"])
        self.assertEqual(sha256(raw), source["sha256"])
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        self.assertEqual(len(text), source["canonical_chars"])
        self.assertEqual(sha256(text.encode()), source["canonical_sha256"])
        for key, span in self.dataset["evidence"].items():
            if span["source_id"] == source["id"]:
                with self.subTest(span=key):
                    self.assertEqual(
                        sha256(text[span["start"] : span["end"]].encode()),
                        span["sha256"],
                    )
                    lines = [
                        text.count("\n", 0, span["start"]) + 1,
                        text.count("\n", 0, span["end"] - 1) + 1,
                    ]
                    self.assertEqual(lines, span["lines"])

    def test_synthetic_source_bytes_and_spans(self):
        for source in self.sources.values():
            if source["kind"] == "fixture":
                with self.subTest(source=source["id"]):
                    self.check_source_bytes_and_spans(source)

    @unittest.skipUnless(
        os.environ.get("ARM64_VERIFY_UPSTREAM") == "1", "Optional upstream check"
    )
    def test_pinned_upstream_bytes_licenses_and_spans(self):
        for source in self.sources.values():
            if source["kind"] == "upstream":
                with self.subTest(source=source["id"]):
                    self.check_source_bytes_and_spans(source)
                    license_record = self.manifest["licenses"][source["license_id"]]
                    self.assertEqual(
                        sha256((ROOT / license_record["local_path"]).read_bytes()),
                        license_record["sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
