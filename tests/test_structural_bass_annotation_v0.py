from __future__ import annotations

import csv
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.serve_structural_bass_annotation import create_server


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/structural_bass_annotation_v0"
FORBIDDEN_BLIND_KEYS = {
    "R_B", "N_B", "next_low_gap_sec", "local_median_ioi", "R_stratum",
    "R_stratum_empirical_min", "R_stratum_empirical_max", "pitch_band",
    "onset_group_index", "original_candidate_identifier",
}
LABELS = {"STRUCTURAL_BASS", "NOT_STRUCTURAL_BASS", "AMBIGUOUS"}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def keys_recursive(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from keys_recursive(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys_recursive(child)


class StructuralBassAnnotationDataTest(unittest.TestCase):
    def test_split_and_hidden_master_invariants(self):
        master = csv_rows(OUTPUT / "annotation_master_hidden.csv")
        manifest = csv_rows(OUTPUT / "annotation_split_manifest.csv")
        self.assertEqual(len(master), 100)
        self.assertEqual(len(manifest), 100)
        self.assertEqual(len({row["original_candidate_identifier"] for row in master}), 100)
        a_ids = {row["original_candidate_identifier"] for row in master if row["set"] == "A"}
        b_ids = {row["original_candidate_identifier"] for row in master if row["set"] == "B"}
        self.assertEqual(len(a_ids), 50)
        self.assertEqual(len(b_ids), 50)
        self.assertFalse(a_ids & b_ids)
        for set_name in ("A", "B"):
            rows = [row for row in manifest if row["set"] == set_name]
            pieces = sorted({row["piece_id"] for row in rows})
            self.assertEqual(len(pieces), 5)
            for piece_id in pieces:
                piece_rows = [row for row in rows if row["piece_id"] == piece_id]
                self.assertEqual(len(piece_rows), 10)
                for stratum in {row["original_R_stratum"] for row in piece_rows}:
                    self.assertEqual(sum(row["original_R_stratum"] == stratum for row in piece_rows), 2)
            self.assertEqual([row["review_id"] for row in rows], [f"{set_name}-{index:03d}" for index in range(1, 51)])
        for key in ("R_B", "N_B", "next_low_gap_sec", "local_median_ioi", "original_R_stratum", "pitch_band"):
            self.assertTrue(all(row[key] != "" for row in master), key)

    def test_blind_payload_has_only_midi_onset_pitch_context(self):
        all_review_ids = set()
        for set_name in ("A", "B"):
            payload = json.loads((OUTPUT / "public" / f"set_{set_name}_blind.json").read_text(encoding="utf-8"))
            rows = payload["candidates"]
            self.assertEqual(len(rows), 50)
            self.assertTrue(all(row["set"] == set_name for row in rows))
            self.assertFalse(FORBIDDEN_BLIND_KEYS & set(keys_recursive(payload)))
            for row in rows:
                self.assertEqual(len(row["context"]), 13)
                self.assertEqual([group["relative_group"] for group in row["context"]], list(range(-4, 9)))
                self.assertTrue(all(group["pitches"] for group in row["context"]))
                center = row["context"][4]
                self.assertEqual(center["relative_time_sec"], 0.0)
                self.assertIn(row["lowest_pitch"], center["pitches"])
                self.assertNotIn(row["review_id"], all_review_ids)
                all_review_ids.add(row["review_id"])
        self.assertEqual(len(all_review_ids), 100)

    def test_initial_results_blank_and_cpu_only_metadata(self):
        rows = csv_rows(OUTPUT / "annotation_results_template.csv")
        current = csv_rows(OUTPUT / "annotation_results.csv")
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows, current)
        self.assertTrue(all(not row["annotation"] and not row["comment"] for row in rows))
        metadata = json.loads((OUTPUT / "build_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["asap_test_midi_access_count"], 0)
        self.assertEqual(metadata["asap_midi_files_opened"], 0)
        self.assertEqual(metadata["cc64_features_used"], 0)
        self.assertEqual(metadata["key_off_features_used"], 0)
        self.assertEqual(metadata["duration_features_used"], 0)
        self.assertFalse(metadata["audio_used"])
        self.assertFalse(metadata["gpu_cuda_used"])

    def test_page_contains_controls_shortcuts_navigation_and_visualization(self):
        html = (OUTPUT / "public/index.html").read_text(encoding="utf-8")
        javascript = (OUTPUT / "public/app.js").read_text(encoding="utf-8")
        for label in ("Structural Bass", "Not Structural Bass", "Ambiguous"):
            self.assertIn(label, html)
        self.assertIn("Pedal 사용의 적절성을 평가하는 단계가 아닙니다", html)
        self.assertIn("renderPianoRoll", javascript)
        self.assertIn('"1": () => labelCurrent("STRUCTURAL_BASS")', javascript)
        self.assertIn('"2": () => labelCurrent("NOT_STRUCTURAL_BASS")', javascript)
        self.assertIn('"3": () => labelCurrent("AMBIGUOUS")', javascript)
        self.assertIn("ArrowLeft: () => navigate(-1)", javascript)
        self.assertIn("ArrowRight: () => navigate(1)", javascript)
        self.assertIn("await saveResult", javascript)
        self.assertIn("state.index += 1", javascript)
        self.assertNotIn("R_B", html + javascript)
        self.assertNotIn("N_B", html + javascript)
        self.assertNotIn("next_low", html + javascript)
        self.assertNotIn("local_median", html + javascript)
        self.assertNotIn("stratum", html + javascript.lower())


class StructuralBassAnnotationServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp.name) / "annotation"
        shutil.copytree(OUTPUT, self.data_root)
        self.server = create_server(self.data_root, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def get(self, path: str):
        with urlopen(self.base + path, timeout=3) as response:
            return response.status, response.headers, response.read()

    def post(self, payload):
        request = Request(
            self.base + "/api/annotation",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read())

    def test_health_static_and_hidden_files_not_served(self):
        status, _, body = self.get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["candidate_count"], 100)
        status, _, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Structural Bass Annotation", body)
        for hidden in ("annotation_master_hidden.csv", "annotation_split_manifest.csv", "annotation_results.csv"):
            with self.assertRaises(HTTPError) as caught:
                self.get("/" + hidden)
            self.assertEqual(caught.exception.code, 404)

    def test_save_refresh_modify_and_export(self):
        first = self.post({
            "review_id": "A-001",
            "annotation": "STRUCTURAL_BASS",
            "comment": "first pass",
        })
        self.assertEqual(first["annotation"], "STRUCTURAL_BASS")
        _, _, body = self.get("/api/results?set=A")
        refreshed = json.loads(body)["results"]
        saved = next(row for row in refreshed if row["review_id"] == "A-001")
        self.assertEqual(saved["annotation"], "STRUCTURAL_BASS")
        self.assertEqual(saved["comment"], "first pass")

        modified = self.post({
            "review_id": "A-001",
            "annotation": "AMBIGUOUS",
            "comment": "revised",
        })
        self.assertEqual(modified["annotation"], "AMBIGUOUS")
        persisted_rows = csv_rows(self.data_root / "annotation_results.csv")
        persisted = next(row for row in persisted_rows if row["review_id"] == "A-001")
        self.assertEqual(persisted["annotation"], "AMBIGUOUS")
        self.assertEqual(persisted["comment"], "revised")

        _, headers, exported = self.get("/api/export?set=A")
        self.assertIn("structural_bass_annotations_A.csv", headers["Content-Disposition"])
        export_rows = list(csv.DictReader(io_text(exported)))
        self.assertEqual(len(export_rows), 50)
        self.assertEqual(tuple(export_rows[0]), ("review_id", "set", "annotation", "comment"))

    def test_reject_invalid_label(self):
        with self.assertRaises(HTTPError) as caught:
            self.post({"review_id": "A-001", "annotation": "AUTO_LABEL", "comment": ""})
        self.assertEqual(caught.exception.code, 400)


def io_text(value: bytes):
    import io
    return io.StringIO(value.decode("utf-8-sig"), newline="")


if __name__ == "__main__":
    unittest.main()
