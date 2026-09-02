"""Strict, atomic run-status persistence for Custom Event Model training."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class NonFiniteStatusMetricError(ValueError):
    """A monitoring value that must be finite is not finite."""

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = fields
        joined = ", ".join(f"{row['path']}={row['kind']}" for row in fields)
        super().__init__("unexpected non-finite status metric: " + joined)


def _kind(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "positive_infinity" if value > 0 else "negative_infinity"


def _walk_nonfinite(value: Any, path: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append({"path": path or "$", "kind": _kind(value)})
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_walk_nonfinite(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_walk_nonfinite(child, f"{path}[{index}]"))
    return found


def _replace_path(root: dict[str, Any], path: str, replacement: Any) -> None:
    # Status schema currently has no list-valued metric path. Keep this helper
    # intentionally strict so schema growth cannot silently sanitize data.
    if "[" in path:
        raise ValueError(f"unsupported non-finite list path: {path}")
    keys = path.split(".")
    target: Any = root
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = replacement


def status_to_json_safe(
    status: Mapping[str, Any], *, failure_mode: bool = False
) -> dict[str, Any]:
    """Validate the status schema and return a strict-JSON-safe copy.

    Only two non-finite cases have special meaning:

    * legacy pre-validation ``best_val_total=+inf`` with no best epoch is an
      undefined sentinel and becomes JSON null;
    * an AMP overflow's non-finite gradient norm is the event being reported;
      it becomes null with an explicit classification field.

    Every other non-finite value raises, unless ``failure_mode`` is active. In
    failure mode it is made explicit in ``nonfinite_status_fields`` so a
    primary numerical exception can still be persisted without being masked.
    """

    result = copy.deepcopy(dict(status))
    best = result.get("best_val_total")
    if (
        isinstance(best, float)
        and math.isinf(best)
        and best > 0
        and result.get("best_epoch") is None
        and result.get("latest_val_total") is None
    ):
        result["best_val_total"] = None
        undefined = list(result.get("undefined_status_fields", []))
        if "best_val_total" not in undefined:
            undefined.append("best_val_total")
        result["undefined_status_fields"] = undefined

    overflow = result.get("last_overflow")
    if isinstance(overflow, dict):
        gradient_norm = overflow.get("gradient_norm")
        if isinstance(gradient_norm, float) and not math.isfinite(gradient_norm):
            overflow["gradient_norm"] = None
            overflow["gradient_norm_is_finite"] = False
            overflow["gradient_norm_nonfinite_kind"] = _kind(gradient_norm)

    nonfinite = _walk_nonfinite(result)
    if nonfinite and not failure_mode:
        raise NonFiniteStatusMetricError(nonfinite)
    if nonfinite:
        for row in nonfinite:
            _replace_path(result, row["path"], None)
        result["nonfinite_status_fields"] = nonfinite
    # A final strict encoding check catches unsupported objects before any file
    # is opened by the atomic writer.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def atomic_write_strict_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """fsync a complete strict JSON temporary file, then atomically replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_run_status(path: str | Path, status: Mapping[str, Any]) -> None:
    atomic_write_strict_json(path, status_to_json_safe(status))


def write_failure_status(
    path: str | Path,
    status: Mapping[str, Any],
    *,
    failure_reason: str,
    primary_traceback: str,
    updated_at: str,
    emergency_log_path: str | Path,
) -> Exception | None:
    """Persist a failed state without replacing the primary exception.

    Returns the secondary logging exception, if any. The caller must re-raise
    its original exception with a bare ``raise``.
    """

    failed = copy.deepcopy(dict(status))
    failed.update(
        status="failed",
        stage="failed",
        failure_reason=failure_reason,
        traceback=primary_traceback,
        updated_at=updated_at,
    )
    try:
        atomic_write_strict_json(path, status_to_json_safe(failed, failure_mode=True))
        return None
    except Exception as logging_error:  # last-resort provenance, never re-raise here
        emergency = Path(emergency_log_path)
        emergency.parent.mkdir(parents=True, exist_ok=True)
        try:
            with emergency.open("a", encoding="utf-8") as handle:
                handle.write("PRIMARY FAILURE\n")
                handle.write(primary_traceback)
                handle.write("\nSTATUS-WRITE FAILURE\n")
                handle.write(repr(logging_error) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            pass
        return logging_error

