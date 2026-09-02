#!/usr/bin/env python3
"""Serve the Structural Bass blind annotation page with CSV persistence."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "analysis/structural_bass_annotation_v0"
SETS = ("A", "B")
LABELS = ("STRUCTURAL_BASS", "NOT_STRUCTURAL_BASS", "AMBIGUOUS")
RESULT_FIELDS = ("review_id", "set", "annotation", "comment")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class AnnotationStore:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self.public_root = (self.data_root / "public").resolve()
        self.results_path = self.data_root / "annotation_results.csv"
        self.lock = threading.RLock()
        self.candidates: dict[str, dict[str, Any]] = {}
        self.valid_ids: dict[str, str] = {}
        for set_name in SETS:
            path = self.public_root / f"set_{set_name}_blind.json"
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("candidates", [])
            if payload.get("set") != set_name or len(rows) != 50:
                raise RuntimeError(f"invalid blind payload: {path}")
            for row in rows:
                review_id = str(row["review_id"])
                if review_id in self.valid_ids or row.get("set") != set_name:
                    raise RuntimeError(f"duplicate/mismatched review ID: {review_id}")
                self.valid_ids[review_id] = set_name
            self.candidates[set_name] = payload
        if not self.results_path.is_file():
            self._write_rows([
                {"review_id": review_id, "set": set_name, "annotation": "", "comment": ""}
                for review_id, set_name in sorted(self.valid_ids.items())
            ])
        self._validate_rows(self._read_rows())

    def _read_rows(self) -> list[dict[str, str]]:
        with self.results_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return [{field: row.get(field, "") for field in RESULT_FIELDS} for row in rows]

    def _validate_rows(self, rows: Sequence[Mapping[str, str]]) -> None:
        if len(rows) != 100 or {row["review_id"] for row in rows} != set(self.valid_ids):
            raise RuntimeError("annotation_results.csv must cover exactly the 100 blind review IDs")
        for row in rows:
            if row["set"] != self.valid_ids[row["review_id"]]:
                raise RuntimeError(f"result set mismatch: {row['review_id']}")
            if row["annotation"] and row["annotation"] not in LABELS:
                raise RuntimeError(f"invalid saved annotation: {row['annotation']}")

    def _write_rows(self, rows: Sequence[Mapping[str, str]]) -> None:
        temporary = self.results_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.results_path)

    def get_candidates(self, set_name: str) -> Mapping[str, Any]:
        return self.candidates[set_name]

    def get_results(self, set_name: str | None = None) -> list[dict[str, str]]:
        with self.lock:
            rows = self._read_rows()
        return [row for row in rows if set_name is None or row["set"] == set_name]

    def update(self, review_id: str, annotation: str, comment: str) -> dict[str, str]:
        if review_id not in self.valid_ids:
            raise ValueError("unknown review_id")
        if annotation not in ("", *LABELS):
            raise ValueError("invalid annotation")
        if len(comment) > 4000:
            raise ValueError("comment exceeds 4000 characters")
        with self.lock:
            rows = self._read_rows()
            result: dict[str, str] | None = None
            for row in rows:
                if row["review_id"] == review_id:
                    row["annotation"] = annotation
                    row["comment"] = comment
                    result = dict(row)
                    break
            if result is None:
                raise RuntimeError("review ID disappeared from results")
            self._validate_rows(rows)
            self._write_rows(rows)
            return result

    def export_csv(self, set_name: str | None = None) -> bytes:
        rows = self.get_results(set_name)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode("utf-8-sig")


def make_handler(store: AnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StructuralBassAnnotation/0"

        def log_message(self, format_: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {format_ % args}")

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            extra_headers: Mapping[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def _send_error_json(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message}, status)

        @staticmethod
        def _set_parameter(query: str, required: bool = True) -> str | None:
            values = parse_qs(query).get("set", [])
            if not values and not required:
                return None
            if len(values) != 1 or values[0] not in SETS:
                raise ValueError("set must be A or B")
            return values[0]

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            try:
                if parsed.path in STATIC_FILES:
                    filename, content_type = STATIC_FILES[parsed.path]
                    path = store.public_root / filename
                    self._send_bytes(path.read_bytes(), content_type)
                    return
                if parsed.path == "/healthz":
                    self._send_json({"ok": True, "candidate_count": len(store.valid_ids)})
                    return
                if parsed.path == "/api/candidates":
                    set_name = self._set_parameter(parsed.query)
                    self._send_json(store.get_candidates(set_name))
                    return
                if parsed.path == "/api/results":
                    set_name = self._set_parameter(parsed.query)
                    self._send_json({"set": set_name, "results": store.get_results(set_name)})
                    return
                if parsed.path == "/api/export":
                    set_name = self._set_parameter(parsed.query, required=False)
                    suffix = set_name or "all"
                    body = store.export_csv(set_name)
                    self._send_bytes(
                        body,
                        "text/csv; charset=utf-8",
                        extra_headers={"Content-Disposition": f'attachment; filename="structural_bass_annotations_{suffix}.csv"'},
                    )
                    return
                # Deliberately no generic file serving: hidden manifests and
                # persistent results are unreachable even if their names are guessed.
                self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            except ValueError as error:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:  # avoid leaking local paths/details
                print(f"GET failed: {error!r}")
                self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotation":
                self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                allowed_keys = {"review_id", "annotation", "comment"}
                if set(payload) != allowed_keys:
                    raise ValueError("request fields must be review_id, annotation, comment")
                review_id = payload["review_id"]
                annotation = payload["annotation"]
                comment = payload["comment"]
                if not all(isinstance(value, str) for value in (review_id, annotation, comment)):
                    raise ValueError("all request fields must be strings")
                self._send_json(store.update(review_id, annotation, comment))
            except (ValueError, json.JSONDecodeError) as error:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:
                print(f"POST failed: {error!r}")
                self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

    return Handler


def create_server(data_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    store = AnnotationStore(data_root)
    return ThreadingHTTPServer((host, port), make_handler(store))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = create_server(args.data_root, args.host, args.port)
    host, port = server.server_address[:2]
    print(f"Structural Bass annotation server: http://{host}:{port}", flush=True)
    print(f"Persistent results: {args.data_root.resolve() / 'annotation_results.csv'}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
