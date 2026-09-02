#!/usr/bin/env python3
"""Render locked Stage-2 listening pairs without running model inference.

This runner is intentionally self-contained: it selects the six pieces, proves
their historical test-score numbering, audits old/current Original PT MIDI,
renders only the required locked MIDI files, validates WAVs, and writes all
manifests/report/status artifacts before exiting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import traceback
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mido

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.listening_comparison_5class_v0 import (
    CHANNELS,
    DEFAULT_INSTRUMENT,
    DEFAULT_RENDERER,
    SAMPLE_RATE,
    _render_command,
    render_midi,
)


TEST_ROOT = REPO / "analysis/stage2_binary_v0/test_eval_v0"
OUTPUT_ROOT = REPO / "analysis/stage2_binary_v0/listening_render_v0"
PIECE_METRICS = TEST_ROOT / "piece_level_metrics.csv"
FINAL_TEST_REPORT = TEST_ROOT / "FINAL_TEST_REPORT.md"
TEST_SCORE_MANIFEST = TEST_ROOT / "test_score_manifest.csv"
FINAL_TEST_STATUS = TEST_ROOT / "run_status.json"
SCORE_DIR = REPO / "third_party/PianistTransformer/data/midis/testset/score"
STAGE1_ROOT = TEST_ROOT / "stage1_cache"
INDEPENDENT_ROOT = TEST_ROOT / "candidate_midis/independent_4x2"
OLD_MIDI_ROOT = REPO / "outputs/midi"
OLD_AUDIO_ROOT = REPO / "outputs/audio"
OLD_RENDER_LOG_ROOT = REPO / "logs/listening_comparison_5class_all_v0"
RENDER_SOURCE = REPO / "src/listening_comparison_5class_v0.py"

SESSION_NAME = "stage2-binary-listening-render-v0"
EPS = 1e-15


class StageError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    if not path.is_file():
        raise StageError(f"missing {label}: {path}")
    if nonempty and path.stat().st_size == 0:
        raise StageError(f"empty {label}: {path}")
    return path


def parse_piece_selection() -> list[dict[str, Any]]:
    require_file(PIECE_METRICS, "piece-level metrics")
    rows = read_csv(PIECE_METRICS)
    if len(rows) != 23 or len({row["piece_id"] for row in rows}) != 23:
        raise StageError(
            f"piece_level_metrics must contain 23 unique pieces, found {len(rows)} rows"
        )
    parsed: list[dict[str, Any]] = []
    for row in rows:
        original = float(row["original_pt_js_distance"])
        independent = float(row["independent_js_distance"])
        improvement = original - independent
        recorded = float(row["independent_js_improvement_vs_original_pt"])
        if not math.isclose(improvement, recorded, rel_tol=0.0, abs_tol=EPS):
            raise StageError(
                f"piece metric arithmetic mismatch for {row['piece_id']}: "
                f"computed={improvement}, recorded={recorded}"
            )
        parsed.append(
            {
                "piece_id": row["piece_id"],
                "composer": row["composer"],
                "title": row["title"],
                "original_pt_js": original,
                "independent_js": independent,
                "js_improvement": improvement,
            }
        )

    top = sorted(parsed, key=lambda item: (-item["js_improvement"], item["piece_id"]))[:4]
    bottom = sorted(parsed, key=lambda item: (item["js_improvement"], item["piece_id"]))[:2]
    for item in top:
        item["category"] = "top4_improved"
    for item in bottom:
        item["category"] = "bottom2_degraded"
    selected = top + bottom
    if len(selected) != 6 or len({item["piece_id"] for item in selected}) != 6:
        raise StageError("top-4/bottom-2 selection did not yield six unique pieces")
    return selected


def numbered_score_hashes() -> dict[str, int]:
    mapping: dict[str, int] = {}
    numbers: set[int] = set()
    for path in SCORE_DIR.glob("*.mid"):
        if not path.stem.isdigit():
            continue
        number = int(path.stem)
        digest = sha256_file(path)
        if digest in mapping:
            raise StageError(f"ambiguous duplicate canonical score hash: {digest}")
        mapping[digest] = number
        numbers.add(number)
    if numbers != set(range(23)):
        raise StageError(f"expected canonical score numbers 0..22, found {sorted(numbers)}")
    return mapping


def attach_score_numbers(selected: list[dict[str, Any]]) -> None:
    manifest_rows = read_csv(require_file(TEST_SCORE_MANIFEST, "test score manifest"))
    if len(manifest_rows) != 23:
        raise StageError(f"expected 23 locked test-score rows, found {len(manifest_rows)}")
    manifest_by_piece = {row["piece_id"]: row for row in manifest_rows}
    if len(manifest_by_piece) != 23:
        raise StageError("locked test-score manifest contains duplicate piece_id values")
    number_by_hash = numbered_score_hashes()

    for item in selected:
        row = manifest_by_piece.get(item["piece_id"])
        if row is None:
            raise StageError(f"selected piece missing from locked score manifest: {item['piece_id']}")
        recorded_hash = row["selected_score_sha256"]
        score_path = Path(row["selected_score_absolute_path"])
        require_file(score_path, "locked selected score")
        actual_hash = sha256_file(score_path)
        if actual_hash != recorded_hash:
            raise StageError(f"locked selected-score hash mismatch: {score_path}")
        if recorded_hash not in number_by_hash:
            raise StageError(
                f"no exact canonical numbered score hash for {item['piece_id']}: {recorded_hash}"
            )
        number = number_by_hash[recorded_hash]
        canonical_score = SCORE_DIR / f"{number}.mid"
        if sha256_file(canonical_score) != recorded_hash:
            raise StageError(f"canonical score hash changed: {canonical_score}")
        item.update(
            {
                "num": number,
                "selected_score_path": str(score_path),
                "score_sha256": recorded_hash,
                "canonical_numbered_score": str(canonical_score),
            }
        )
    if len({item["num"] for item in selected}) != 6:
        raise StageError("score hash mapping did not yield six unique test-score numbers")


def selected_paths(item: dict[str, Any]) -> dict[str, Path]:
    piece_id = item["piece_id"]
    number = item["num"]
    return {
        "current_original": STAGE1_ROOT / piece_id / "original_pt.mid",
        "independent": INDEPENDENT_ROOT / f"{piece_id}.mid",
        "old_original": OLD_MIDI_ROOT / f"{number}_original_pt.mid",
        "old_audio": OLD_AUDIO_ROOT / f"{number}_original_pt.wav",
        "developed_audio": OLD_AUDIO_ROOT / f"{number}_developed_PT.wav",
        "fallback_audio": OLD_AUDIO_ROOT / f"{number}_original_pt_locked_test.wav",
        "prior_render_log": OLD_RENDER_LOG_ROOT / f"{number}.stdout.json",
    }


def verify_input_paths(selected: list[dict[str, Any]]) -> None:
    require_file(FINAL_TEST_REPORT, "final test report")
    status = json.loads(require_file(FINAL_TEST_STATUS, "final test status").read_text())
    if status.get("status") != "completed":
        raise StageError(f"locked final test is not completed: {status.get('status')}")
    for item in selected:
        paths = selected_paths(item)
        require_file(paths["current_original"], "current locked Original PT MIDI")
        require_file(paths["independent"], "locked independent MIDI")
        require_file(paths["old_original"], "old Original PT MIDI")
        require_file(paths["old_audio"], "old Original PT audio")
        require_file(paths["prior_render_log"], "historical renderer log")


def renderer_provenance(selected: list[dict[str, Any]]) -> dict[str, Any]:
    renderer = require_file(Path(DEFAULT_RENDERER), "sfizz renderer")
    instrument = require_file(Path(DEFAULT_INSTRUMENT), "Salamander SFZ")
    require_file(RENDER_SOURCE, "historical renderer source")
    if not os.access(renderer, os.X_OK):
        raise StageError(f"renderer is not executable: {renderer}")

    verified_logs: list[str] = []
    for item in selected:
        paths = selected_paths(item)
        payload = json.loads(paths["prior_render_log"].read_text(encoding="utf-8"))
        commands = payload.get("audio", {}).get("render_commands", [])
        expected = _render_command(renderer, instrument, paths["old_original"], paths["old_audio"])
        if expected not in commands:
            raise StageError(
                f"historical render command does not prove the locked renderer settings for num={item['num']}"
            )
        audio = payload.get("audio", {})
        if int(audio.get("sample_rate", -1)) != SAMPLE_RATE or int(audio.get("channels", -1)) != CHANNELS:
            raise StageError(f"historical WAV settings mismatch for num={item['num']}")
        verified_logs.append(str(paths["prior_render_log"]))

    return {
        "renderer": str(renderer),
        "renderer_sha256": sha256_file(renderer),
        "instrument": str(instrument),
        "instrument_sfz_sha256": sha256_file(instrument),
        "renderer_source": str(RENDER_SOURCE),
        "renderer_source_sha256": sha256_file(RENDER_SOURCE),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "gain": "sfizz_render default (no gain option)",
        "tail_and_duration": "--use-eot",
        "post_processing": "none",
        "historical_logs_verified": verified_logs,
    }


def normalize_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return list(value)
    if isinstance(value, tuple):
        return [normalize_json(item) for item in value]
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in sorted(value.items())}
    return value


def midi_semantic_signature(path: Path) -> dict[str, Any]:
    """Represent every ordered MIDI/meta event at its absolute tick.

    This is intentionally stricter than a note-count comparison. It preserves
    track structure and ordering and includes EOT timing because the historical
    renderer uses --use-eot.
    """
    midi = mido.MidiFile(str(path), clip=False)
    tracks: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        messages: list[dict[str, Any]] = []
        for order, message in enumerate(track):
            absolute_tick += int(message.time)
            data = message.dict()
            data.pop("time", None)
            messages.append(
                {
                    "order": order,
                    "absolute_tick": absolute_tick,
                    "message": normalize_json(data),
                }
            )
        tracks.append({"track_index": track_index, "messages": messages})
    return {
        "type": midi.type,
        "ticks_per_beat": midi.ticks_per_beat,
        "tracks": tracks,
    }


def signature_summary(signature: dict[str, Any]) -> dict[str, Any]:
    message_types: Counter[str] = Counter()
    note_on = 0
    note_off = 0
    cc64 = 0
    other_cc = 0
    for track in signature["tracks"]:
        for event in track["messages"]:
            message = event["message"]
            kind = str(message.get("type"))
            message_types[kind] += 1
            if kind == "note_on" and int(message.get("velocity", 0)) > 0:
                note_on += 1
            elif kind == "note_off" or (kind == "note_on" and int(message.get("velocity", 0)) == 0):
                note_off += 1
            elif kind == "control_change":
                if int(message.get("control", -1)) == 64:
                    cc64 += 1
                else:
                    other_cc += 1
    return {
        "midi_type": signature["type"],
        "ticks_per_beat": signature["ticks_per_beat"],
        "track_count": len(signature["tracks"]),
        "note_on_count": note_on,
        "note_off_count": note_off,
        "cc64_count": cc64,
        "other_cc_count": other_cc,
        "message_type_counts": dict(sorted(message_types.items())),
    }


def equality_audit(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        paths = selected_paths(item)
        current_hash = sha256_file(paths["current_original"])
        old_hash = sha256_file(paths["old_original"])
        current_signature = midi_semantic_signature(paths["current_original"])
        old_signature = midi_semantic_signature(paths["old_original"])
        semantic_equal = current_signature == old_signature
        if current_hash == old_hash:
            status = "BYTE_IDENTICAL"
            if not semantic_equal:
                raise StageError(f"byte-identical MIDI parsed differently for {item['piece_id']}")
        elif semantic_equal:
            status = "SEMANTICALLY_IDENTICAL"
        else:
            status = "DIFFERENT"

        reuse = status in {"BYTE_IDENTICAL", "SEMANTICALLY_IDENTICAL"} and paths["old_audio"].is_file()
        actual_baseline = paths["old_audio"] if reuse else paths["fallback_audio"]
        rows.append(
            {
                "num": item["num"],
                "piece_id": item["piece_id"],
                "current_original_pt_midi": str(paths["current_original"]),
                "current_original_pt_sha256": current_hash,
                "old_original_pt_midi": str(paths["old_original"]),
                "old_original_pt_sha256": old_hash,
                "equality_status": status,
                "semantic_signature_equal": semantic_equal,
                "current_semantic_summary": json.dumps(signature_summary(current_signature), sort_keys=True),
                "old_semantic_summary": json.dumps(signature_summary(old_signature), sort_keys=True),
                "old_original_pt_audio": str(paths["old_audio"]),
                "baseline_audio_reused": reuse,
                "actual_baseline_audio_path": str(actual_baseline),
            }
        )
    return rows


def preserve_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    digest = sha256_file(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    preserved = path.with_name(f"{path.stem}.preexisting_{stamp}_{digest[:12]}{path.suffix}")
    if preserved.exists():
        raise StageError(f"cannot preserve existing output; backup already exists: {preserved}")
    path.replace(preserved)
    return preserved


def inspect_wav(path: Path) -> dict[str, Any]:
    require_file(path, "rendered WAV")
    nonzero = False
    with wave.open(str(path), "rb") as handle:
        metadata = {
            "sample_rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "frames": handle.getnframes(),
            "duration_seconds": handle.getnframes() / handle.getframerate(),
        }
        while True:
            block = handle.readframes(65536)
            if not block:
                break
            if any(block):
                nonzero = True
                break
    if metadata["sample_rate"] != SAMPLE_RATE:
        raise StageError(f"unexpected WAV sample rate for {path}: {metadata['sample_rate']}")
    if metadata["channels"] != CHANNELS:
        raise StageError(f"unexpected WAV channels for {path}: {metadata['channels']}")
    if metadata["duration_seconds"] <= 0.5 or metadata["frames"] <= 0:
        raise StageError(f"empty or implausibly short WAV: {path}")
    if not nonzero:
        raise StageError(f"WAV has no non-zero PCM samples: {path}")
    return metadata


def validate_duration_pair(baseline: dict[str, Any], developed: dict[str, Any]) -> None:
    durations = [baseline["duration_seconds"], developed["duration_seconds"]]
    if min(durations) <= 0.5 or abs(durations[0] - durations[1]) > max(5.0, 0.25 * max(durations)):
        raise StageError(f"WAV durations are unreasonable: {durations}")


def selected_csv_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "num": item["num"],
            "piece_id": item["piece_id"],
            "composer": item["composer"],
            "title": item["title"],
            "original_pt_js": item["original_pt_js"],
            "independent_js": item["independent_js"],
            "js_improvement": item["js_improvement"],
            "category": item["category"],
        }
        for item in selected
    ]


def setup_logging(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("stage2_binary_listening_render")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_root / "render.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.handlers[:] = [file_handler, stream_handler]
    return logger


def set_owner(paths: Iterable[Path]) -> None:
    owner = REPO.stat()
    for root in paths:
        if not root.exists():
            continue
        candidates = [root]
        if root.is_dir():
            candidates.extend(root.rglob("*"))
        for path in candidates:
            try:
                os.chown(path, owner.st_uid, owner.st_gid)
            except PermissionError:
                pass


def write_report(
    selected: list[dict[str, Any]],
    equality_rows: list[dict[str, Any]],
    render_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    equal_by_piece = {row["piece_id"]: row for row in equality_rows}
    render_by_piece = {row["piece_id"]: row for row in render_rows}
    lines = [
        "# Stage 2 Binary Locked Test Listening Render Report",
        "",
        "## Scope",
        "",
        "This run generated listening artifacts only. It performed no model inference, training, checkpoint selection, threshold/decoding change, or metric recomputation.",
        "",
        "## Selected pieces",
        "",
        "Selection was deterministic from locked final-test piece metrics: `Original_PT_piece_JS - independent_4x2_piece_JS`; ties use lexical `piece_id` order.",
        "",
        "| num | Piece | category | Original PT JS | independent JS | JS improvement |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in selected:
        label = f"{item['composer']} — {item['title']} (`{item['piece_id']}`)"
        lines.append(
            f"| {item['num']} | {label} | {item['category']} | "
            f"{item['original_pt_js']:.12f} | {item['independent_js']:.12f} | "
            f"{item['js_improvement']:+.12f} |"
        )
    counts = Counter(row["equality_status"] for row in equality_rows)
    reused = sum(bool(row["baseline_audio_reused"]) for row in equality_rows)
    lines.extend(
        [
            "",
            "## Old Original PT reuse audit",
            "",
            f"- BYTE_IDENTICAL: {counts['BYTE_IDENTICAL']}",
            f"- SEMANTICALLY_IDENTICAL: {counts['SEMANTICALLY_IDENTICAL']}",
            f"- DIFFERENT: {counts['DIFFERENT']}",
            f"- Existing Original PT WAV reused: {reused}/6",
            "",
            "The semantic comparison covered ordered tracks and every MIDI/meta event at its absolute tick, including notes, velocity, CC64 and other controls, programs/channels, tempo, time/key signatures, markers/lyrics, pitch bends, and end-of-track timing.",
            "",
            "## Renderer provenance",
            "",
            f"- Existing renderer source: `{provenance['renderer_source']}` (`{provenance['renderer_source_sha256']}`)",
            f"- Executable: `{provenance['renderer']}` (`{provenance['renderer_sha256']}`)",
            f"- Instrument: `{provenance['instrument']}` (`{provenance['instrument_sfz_sha256']}`)",
            f"- Parameters: `{SAMPLE_RATE}` Hz, `{CHANNELS}` channels, default sfizz gain, `--use-eot`, no post-processing",
            f"- Historical command logs verified: {len(provenance['historical_logs_verified'])}/6",
            "",
            "## Listening pairs",
            "",
            "| num | Piece | category | Original PT audio | Developed PT audio |",
            "|---:|---|---|---|---|",
        ]
    )
    for item in selected:
        audit = equal_by_piece[item["piece_id"]]
        render = render_by_piece[item["piece_id"]]
        label = f"{item['composer']} — {item['title']}"
        lines.append(
            f"| {item['num']} | {label} | {item['category']} | "
            f"`{audit['actual_baseline_audio_path']}` | `{render['developed_audio_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- Selected pieces: 6 unique (top four improved + bottom two degraded)",
            "- Test-score number mapping: exact locked score SHA-256 → canonical numbered score SHA-256",
            "- Developed renders: 6/6",
            "- WAV checks: readable PCM, 48 kHz stereo, finite non-zero duration/audio, and no obvious truncation versus baseline",
            "- Joint-16 MIDI was not rendered",
            "- No existing Original PT audio was overwritten",
            "",
        ]
    )
    (OUTPUT_ROOT / "RENDER_LISTENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def static_preflight() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise StageError(f"refusing existing output root: {OUTPUT_ROOT}")
    selected = parse_piece_selection()
    attach_score_numbers(selected)
    verify_input_paths(selected)
    provenance = renderer_provenance(selected)
    return {
        "passed": True,
        "session_name": SESSION_NAME,
        "selected": [
            {
                "num": item["num"],
                "piece_id": item["piece_id"],
                "category": item["category"],
                "js_improvement": item["js_improvement"],
            }
            for item in selected
        ],
        "renderer": provenance,
        "existing_developed_outputs": [
            str(selected_paths(item)["developed_audio"])
            for item in selected
            if selected_paths(item)["developed_audio"].exists()
        ],
    }


def run() -> None:
    if OUTPUT_ROOT.exists():
        raise StageError(f"refusing existing output root: {OUTPUT_ROOT}")
    logger = setup_logging(OUTPUT_ROOT)
    status_path = OUTPUT_ROOT / "run_status.json"
    generated_audio: list[Path] = []
    preserved_audio: list[Path] = []
    stage = "initialize"
    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": stage,
            "started_at": utc_now(),
            "pid": os.getpid(),
            "tmux_session": SESSION_NAME,
        },
    )

    try:
        stage = "select_pieces_and_map_numbers"
        logger.info("stage=%s", stage)
        selected = parse_piece_selection()
        attach_score_numbers(selected)
        verify_input_paths(selected)
        selection_rows = selected_csv_rows(selected)
        atomic_csv(
            OUTPUT_ROOT / "selected_pieces.csv",
            selection_rows,
            ["num", "piece_id", "composer", "title", "original_pt_js", "independent_js", "js_improvement", "category"],
        )
        logger.info("SELECTED_6_READY nums=%s", ",".join(str(item["num"]) for item in selected))

        stage = "verify_renderer_provenance"
        logger.info("stage=%s", stage)
        provenance = renderer_provenance(selected)
        atomic_json(OUTPUT_ROOT / "renderer_provenance.json", provenance)
        logger.info("RENDERER_PROVENANCE_PASS renderer=%s instrument=%s", provenance["renderer"], provenance["instrument"])

        stage = "audit_original_pt_midi_equality"
        logger.info("stage=%s", stage)
        equality_rows = equality_audit(selected)
        atomic_csv(
            OUTPUT_ROOT / "original_pt_reuse_check.csv",
            equality_rows,
            [
                "num", "piece_id", "current_original_pt_midi", "current_original_pt_sha256",
                "old_original_pt_midi", "old_original_pt_sha256", "equality_status",
                "semantic_signature_equal", "current_semantic_summary", "old_semantic_summary",
                "old_original_pt_audio", "baseline_audio_reused", "actual_baseline_audio_path",
            ],
        )
        logger.info(
            "ORIGINAL_PT_EQUALITY_AUDIT_PASS statuses=%s",
            json.dumps(dict(Counter(row["equality_status"] for row in equality_rows)), sort_keys=True),
        )

        equality_by_piece = {row["piece_id"]: row for row in equality_rows}
        render_rows: list[dict[str, Any]] = []
        renderer = Path(provenance["renderer"])
        instrument = Path(provenance["instrument"])

        stage = "render_audio"
        for index, item in enumerate(selected):
            paths = selected_paths(item)
            audit = equality_by_piece[item["piece_id"]]
            preserved = preserve_existing(paths["developed_audio"])
            if preserved is not None:
                preserved_audio.append(preserved)
                logger.warning("preserved unverified existing developed WAV: %s -> %s", paths["developed_audio"], preserved)

            if index == 0:
                atomic_json(
                    status_path,
                    {
                        "status": "running",
                        "stage": "first_render_started",
                        "started_at": utc_now(),
                        "pid": os.getpid(),
                        "tmux_session": SESSION_NAME,
                        "first_render_num": item["num"],
                        "first_render_piece_id": item["piece_id"],
                        "selected_piece_count": 6,
                        "selected_pieces_csv": str(OUTPUT_ROOT / "selected_pieces.csv"),
                        "original_pt_reuse_check_csv": str(OUTPUT_ROOT / "original_pt_reuse_check.csv"),
                        "renderer_provenance": str(OUTPUT_ROOT / "renderer_provenance.json"),
                    },
                )
                logger.info(
                    "FIRST_RENDER_STARTED num=%s piece_id=%s command=%s",
                    item["num"], item["piece_id"], json.dumps(_render_command(renderer, instrument, paths["independent"], paths["developed_audio"])),
                )
                logger.info("INITIAL_READY")
            else:
                logger.info("RENDER_STARTED num=%s piece_id=%s", item["num"], item["piece_id"])

            command = render_midi(renderer, instrument, paths["independent"], paths["developed_audio"])
            generated_audio.append(paths["developed_audio"])
            developed_meta = inspect_wav(paths["developed_audio"])

            fallback_command: list[str] | None = None
            if not bool(audit["baseline_audio_reused"]):
                fallback_preserved = preserve_existing(paths["fallback_audio"])
                if fallback_preserved is not None:
                    preserved_audio.append(fallback_preserved)
                    logger.warning("preserved unverified existing fallback WAV: %s -> %s", paths["fallback_audio"], fallback_preserved)
                logger.info("FALLBACK_BASELINE_RENDER_STARTED num=%s piece_id=%s", item["num"], item["piece_id"])
                fallback_command = render_midi(renderer, instrument, paths["current_original"], paths["fallback_audio"])
                generated_audio.append(paths["fallback_audio"])
                audit["actual_baseline_audio_path"] = str(paths["fallback_audio"])

            baseline_path = Path(audit["actual_baseline_audio_path"])
            baseline_meta = inspect_wav(baseline_path)
            validate_duration_pair(baseline_meta, developed_meta)
            render_rows.append(
                {
                    "num": item["num"],
                    "piece_id": item["piece_id"],
                    "category": item["category"],
                    "independent_midi_path": str(paths["independent"]),
                    "independent_midi_sha256": sha256_file(paths["independent"]),
                    "developed_audio_path": str(paths["developed_audio"]),
                    "developed_audio_sha256": sha256_file(paths["developed_audio"]),
                    "renderer/script provenance": f"{provenance['renderer']} | {provenance['renderer_source']}",
                    "render parameters/config provenance": json.dumps(
                        {
                            "command": command,
                            "fallback_command": fallback_command,
                            "instrument": provenance["instrument"],
                            "sample_rate": SAMPLE_RATE,
                            "channels": CHANNELS,
                            "gain": provenance["gain"],
                            "tail": provenance["tail_and_duration"],
                        }, sort_keys=True,
                    ),
                    "duration": developed_meta["duration_seconds"],
                    "sample_rate": developed_meta["sample_rate"],
                    "channels": developed_meta["channels"],
                    "sample_width_bytes": developed_meta["sample_width_bytes"],
                    "baseline_audio_path": str(baseline_path),
                    "baseline_duration": baseline_meta["duration_seconds"],
                    "duration_difference_seconds": abs(developed_meta["duration_seconds"] - baseline_meta["duration_seconds"]),
                    "preexisting_developed_audio_preserved_path": str(preserved) if preserved else "",
                    "render_status": "PASS",
                }
            )
            logger.info("RENDER_PASS num=%s duration=%.3f", item["num"], developed_meta["duration_seconds"])

        # Re-write after fallback paths, if any, have been finalized.
        atomic_csv(
            OUTPUT_ROOT / "original_pt_reuse_check.csv",
            equality_rows,
            [
                "num", "piece_id", "current_original_pt_midi", "current_original_pt_sha256",
                "old_original_pt_midi", "old_original_pt_sha256", "equality_status",
                "semantic_signature_equal", "current_semantic_summary", "old_semantic_summary",
                "old_original_pt_audio", "baseline_audio_reused", "actual_baseline_audio_path",
            ],
        )
        atomic_csv(
            OUTPUT_ROOT / "render_manifest.csv",
            render_rows,
            [
                "num", "piece_id", "category", "independent_midi_path", "independent_midi_sha256",
                "developed_audio_path", "developed_audio_sha256", "renderer/script provenance",
                "render parameters/config provenance", "duration", "sample_rate", "channels",
                "sample_width_bytes", "baseline_audio_path", "baseline_duration",
                "duration_difference_seconds", "preexisting_developed_audio_preserved_path", "render_status",
            ],
        )

        stage = "write_report_and_verify_artifacts"
        logger.info("stage=%s", stage)
        write_report(selected, equality_rows, render_rows, provenance)
        required = [
            OUTPUT_ROOT / "selected_pieces.csv",
            OUTPUT_ROOT / "original_pt_reuse_check.csv",
            OUTPUT_ROOT / "render_manifest.csv",
            OUTPUT_ROOT / "renderer_provenance.json",
            OUTPUT_ROOT / "render.log",
            OUTPUT_ROOT / "RENDER_LISTENING_REPORT.md",
        ]
        for path in required:
            require_file(path, "required listening artifact")
        if len(read_csv(OUTPUT_ROOT / "selected_pieces.csv")) != 6:
            raise StageError("selected_pieces.csv integrity check failed")
        if len(read_csv(OUTPUT_ROOT / "original_pt_reuse_check.csv")) != 6:
            raise StageError("original_pt_reuse_check.csv integrity check failed")
        manifest = read_csv(OUTPUT_ROOT / "render_manifest.csv")
        if len(manifest) != 6 or any(row["render_status"] != "PASS" for row in manifest):
            raise StageError("render_manifest.csv integrity check failed")
        if len(generated_audio) < 6 or any(not path.is_file() for path in generated_audio):
            raise StageError("generated WAV integrity check failed")

        artifact_hashes = {
            str(path.relative_to(REPO)): sha256_file(path)
            for path in required
            if path.name != "render.log"
        }
        for path in generated_audio:
            artifact_hashes[str(path.relative_to(REPO))] = sha256_file(path)
        atomic_json(OUTPUT_ROOT / "artifact_sha256_manifest.json", artifact_hashes)
        atomic_json(
            status_path,
            {
                "status": "completed",
                "completed_at": utc_now(),
                "selected_piece_count": 6,
                "developed_render_count": 6,
                "fallback_original_render_count": sum(not bool(row["baseline_audio_reused"]) for row in equality_rows),
                "baseline_audio_reused_count": sum(bool(row["baseline_audio_reused"]) for row in equality_rows),
                "report": str(OUTPUT_ROOT / "RENDER_LISTENING_REPORT.md"),
                "artifact_integrity": "PASS",
            },
        )
        logger.info("PIPELINE_COMPLETED report=%s", OUTPUT_ROOT / "RENDER_LISTENING_REPORT.md")
        set_owner([OUTPUT_ROOT, *generated_audio, *preserved_audio])
    except BaseException as exc:
        logger.exception("pipeline failed at stage=%s", stage)
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_stage": stage,
                "timestamp": utc_now(),
                "error_summary": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        set_owner([OUTPUT_ROOT, *generated_audio, *preserved_audio])
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-preflight", action="store_true")
    args = parser.parse_args()
    if args.static_preflight:
        print(json.dumps(static_preflight(), indent=2, sort_keys=True))
        return 0
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
