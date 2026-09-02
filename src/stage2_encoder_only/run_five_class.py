"""Crash-aware end-to-end orchestration for Stage 2 five-class v0.

The entry point deliberately keeps the scientific run in one Python process:
selected synthetic tests, the train-only canonical oracle, the train-only GPU
overfit gate, full training, validation-only comparison, and report writing.
It never constructs an ASAP ``test`` dataset.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import errno
import importlib.util
import hashlib
import io
import json
import math
import os
import pwd
import socket
import subprocess
import sys
import time
import traceback
import unittest
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

from .five_class import CLASS_BOUNDS, CLASS_NAMES, REPRESENTATIVES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "stage2_encoder_only_5class_v0"
DEFAULT_SPLIT_CSV = PROJECT_ROOT / "analysis" / "stage2_encoder_only_v0" / "asap_split.csv"
DEFAULT_TMUX_SESSION = "stage2-5class-v0"
LOCK_NAME = ".run.lock"
STATUS_NAME = "run_status.json"
REPORT_NAME = "FIVE_CLASS_TRAINING_REPORT.md"
KST = timezone(timedelta(hours=9), name="KST")
RUN_STAGES = {
    "initializing",
    "testing",
    "overfit",
    "preloading",
    "training",
    "validation",
    "comparison",
    "reporting",
    "complete",
    "failed",
}
SYNTHETIC_TEST_MODULES = (
    "test_stage2_five_class.py",
    "test_stage2_five_class_runtime.py",
    "test_stage2_ordinal.py",
    "test_stage2_coarse_to_fine.py",
    "test_stage2_calibration.py",
    "test_stage2_posterior.py",
    "test_stage2_model.py",
    "test_stage2_training.py",
    "test_stage2_train.py",
    "test_stage2_evaluate.py",
    "test_stage2_five_class_audit.py",
    "test_stage2_weighted_five_class.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def kst_now() -> str:
    return datetime.now(KST).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError:
        # Some bind/network filesystems do not permit directory fsync. The file
        # itself is still flushed before its atomic rename.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def atomic_text(path: str | Path, text: str) -> None:
    """Write a complete text file through fsync and same-directory rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment_value(
    environ: Mapping[str, str], names: Sequence[str]
) -> tuple[str | None, str | None]:
    for name in names:
        value = environ.get(name)
        if value not in (None, ""):
            return value, name
    return None, None


def _safe_git_query(project_root: Path, arguments: Sequence[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode:
        return None, result.stderr.strip() or f"git exited {result.returncode}"
    return result.stdout.rstrip("\n"), None


def collect_git_metadata(
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect fail-safe Git metadata, preferring host-provided environment."""

    environment = os.environ if environ is None else environ
    root = Path(project_root).resolve()
    warnings: list[str] = []
    commit, commit_source = _environment_value(
        environment,
        ("STAGE2_PROJECT_GIT_COMMIT", "HOST_PROJECT_GIT_COMMIT", "PROJECT_GIT_COMMIT"),
    )
    if commit is None:
        commit, error = _safe_git_query(root, ("rev-parse", "HEAD"))
        commit_source = "git" if commit is not None else "unknown"
        if error:
            warnings.append(f"commit query: {error}")

    dirty_text, dirty_source = _environment_value(
        environment,
        ("STAGE2_PROJECT_GIT_STATUS", "HOST_PROJECT_GIT_STATUS", "PROJECT_GIT_STATUS"),
    )
    if dirty_text is None:
        encoded, encoded_source = _environment_value(
            environment,
            (
                "STAGE2_PROJECT_GIT_STATUS_B64",
                "HOST_PROJECT_GIT_STATUS_B64",
                "PROJECT_GIT_STATUS_B64",
            ),
        )
        if encoded is not None:
            try:
                dirty_text = base64.b64decode(encoded, validate=True).decode("utf-8")
                dirty_source = encoded_source
            except (ValueError, UnicodeDecodeError) as exc:
                warnings.append(f"dirty-status environment decode: {exc}")
    if dirty_text is None:
        dirty_text, error = _safe_git_query(root, ("status", "--short"))
        dirty_source = "git" if dirty_text is not None else "unknown"
        if error:
            warnings.append(f"dirty-status query: {error}")

    return {
        "commit": commit or "unknown",
        "commit_source": commit_source or "unknown",
        "dirty_status": dirty_text if dirty_text is not None else "unknown",
        "dirty": None if dirty_text is None else bool(dirty_text.strip()),
        "dirty_status_source": dirty_source or "unknown",
        "warnings": warnings,
    }


def _read_proc_start_time(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = text.rfind(")")
        if closing < 0:
            return None
        fields_after_command = text[closing + 2 :].split()
        # Field 3 is the first item above; Linux field 22 is therefore index 19.
        return fields_after_command[19]
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        return None


def process_identity(pid: int | None = None) -> dict[str, Any]:
    """Return enough local identity to detect a live process and PID reuse."""

    process_id = os.getpid() if pid is None else int(pid)
    return {
        "pid": process_id,
        "hostname": socket.gethostname(),
        "proc_start_time": _read_proc_start_time(process_id),
    }


def process_identity_is_active(identity: Mapping[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if identity.get("hostname") not in (None, "", socket.gethostname()):
        # A foreign namespace cannot be proven stale from this process.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno != errno.EPERM:
            return False
    recorded_start = identity.get("proc_start_time")
    current_start = _read_proc_start_time(pid)
    if recorded_start and current_start and str(recorded_start) != str(current_start):
        return False
    return True


class DuplicateRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunLock:
    path: Path
    payload: dict[str, Any]
    stale_archives: tuple[str, ...] = ()


def _read_lock_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _lock_is_active(payload: Mapping[str, Any] | None) -> bool:
    if not payload:
        return False
    identity = payload.get("container_process")
    if isinstance(identity, Mapping) and identity.get("hostname") == socket.gethostname():
        return process_identity_is_active(identity)
    host_identity = payload.get("host_process")
    if isinstance(host_identity, Mapping) and host_identity.get("hostname") == socket.gethostname():
        return process_identity_is_active(host_identity)
    # Conservatively retain a lock from an uninspectable container/host.
    return True


def archive_stale_lock(
    lock_path: str | Path,
    payload: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically retain a stale/corrupt lock under a unique audit name."""

    path = Path(lock_path)
    run_fragment = str((payload or {}).get("run_id", "unknown"))[:16]
    archive = path.parent.parent / (
        f"{path.parent.name}.{path.name.lstrip('.')}.stale."
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f".{time.time_ns()}.{run_fragment}"
    )
    os.replace(path, archive)
    _fsync_directory(archive.parent)
    return archive


def _exclusive_link_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fully written lock atomically, failing if the target exists."""

    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{time.time_ns()}.{uuid.uuid4().hex}.candidate"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_run_lock(
    output_dir: str | Path,
    run_id: str,
    host_pid: int | None = None,
    container_pid: int | None = None,
) -> RunLock:
    """Acquire the exclusive active-run lock, archiving proven stale locks."""

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    container_identity = process_identity(container_pid or os.getpid())
    host_identity: dict[str, Any] = {
        "pid": host_pid,
        "hostname": os.environ.get("STAGE2_HOST_HOSTNAME", "unknown"),
        "proc_start_time": os.environ.get("STAGE2_HOST_PROCESS_START"),
    }
    payload = {
        "run_id": run_id,
        "host_pid": host_pid,
        "container_pid": int(container_identity["pid"]),
        "host_process": host_identity,
        "container_process": container_identity,
        "created_at_utc": utc_now(),
        "created_at_kst": kst_now(),
    }
    stale_archives: list[str] = []
    for _ in range(8):
        try:
            _exclusive_link_json(path, payload)
            return RunLock(path=path, payload=payload, stale_archives=tuple(stale_archives))
        except FileExistsError:
            existing = _read_lock_payload(path)
            if _lock_is_active(existing):
                raise DuplicateRunError(
                    f"active five-class run lock already exists: {path}; payload={existing}"
                )
            try:
                stale_archives.append(str(archive_stale_lock(path, existing)))
            except FileNotFoundError:
                continue
    raise RuntimeError(f"could not acquire run lock after stale-lock retries: {path}")


def release_run_lock(lock: RunLock, outcome: str) -> Path | None:
    """Archive this run's lock; never remove or replace another run's lock."""

    current = _read_lock_payload(lock.path)
    if not current or current.get("run_id") != lock.payload.get("run_id"):
        return None
    safe_outcome = "".join(character if character.isalnum() else "_" for character in outcome)
    archive = lock.path.parent.parent / (
        f"{lock.path.parent.name}.{LOCK_NAME.lstrip('.')}.{safe_outcome}."
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f".{time.time_ns()}"
    )
    os.replace(lock.path, archive)
    _fsync_directory(archive.parent)
    return archive


def _parse_optional_pid(value: str | None) -> int | None:
    if value in (None, "", "unknown"):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def initial_run_status(
    run_id: str,
    host_pid: int | None,
    container_pid: int,
    tmux_session: str,
    gpu_uuid: str | None = None,
    start_time_utc: str | None = None,
    start_time_kst: str | None = None,
) -> dict[str, Any]:
    now_utc = start_time_utc or utc_now()
    return {
        "run_id": run_id,
        "stage": "initializing",
        "completed": False,
        "failed": False,
        "current_epoch": 0,
        "current_step": 0,
        "best_epoch": 0,
        "best_validation_loss": None,
        "host_pid": host_pid,
        "container_pid": int(container_pid),
        "tmux_session": tmux_session,
        "gpu_uuid": gpu_uuid or "unknown",
        "start_time_utc": now_utc,
        "start_time_kst": start_time_kst or kst_now(),
        "elapsed_seconds": 0.0,
        "last_update_time": now_utc,
        "failure_stage": None,
        "exception_type": None,
        "exception_message": None,
    }


class AtomicRunStatus:
    """Small state writer that guarantees a complete JSON file per update."""

    def __init__(self, path: str | Path, payload: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.payload = dict(payload)
        self._started = time.monotonic()
        self.write()

    def write(self) -> None:
        atomic_json(self.path, self.payload)

    def update(self, *, stage: str | None = None, **updates: Any) -> dict[str, Any]:
        if stage is not None:
            if stage not in RUN_STAGES:
                raise ValueError(f"unknown run stage: {stage}")
            self.payload["stage"] = stage
        self.payload.update(updates)
        self.payload["elapsed_seconds"] = float(time.monotonic() - self._started)
        self.payload["last_update_time"] = utc_now()
        if self.payload.get("completed") and self.payload.get("failed"):
            raise ValueError("run cannot be both completed and failed")
        self.write()
        return dict(self.payload)

    def fail(self, failure_stage: str, exc: BaseException, **updates: Any) -> dict[str, Any]:
        return self.update(
            stage="failed",
            completed=False,
            failed=True,
            failure_stage=failure_stage,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            **updates,
        )


class Tee:
    def __init__(self, *handles: TextIO) -> None:
        self.handles = handles

    def write(self, text: str) -> int:
        for handle in self.handles:
            handle.write(text)
            handle.flush()
        return len(text)

    def flush(self) -> None:
        for handle in self.handles:
            handle.flush()


def run_selected_synthetic_tests(
    project_root: str | Path,
    output_dir: str | Path,
    modules: Sequence[str] = SYNTHETIC_TEST_MODULES,
) -> dict[str, Any]:
    """Run the audited synthetic allowlist; dataset/test-split smoke is excluded."""

    selected = tuple(modules)
    if not selected or any(module not in SYNTHETIC_TEST_MODULES for module in selected):
        raise ValueError("synthetic test selection contains a non-allowlisted module")
    if any("dataset" in module.lower() for module in selected):
        raise ValueError("dataset test modules are forbidden in this scientific run")
    root = Path(project_root).resolve()
    tests_dir = root / "tests"
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for index, filename in enumerate(selected):
        test_path = tests_dir / filename
        module_name = f"_stage2_five_class_synthetic_{index}_{test_path.stem}"
        specification = importlib.util.spec_from_file_location(module_name, test_path)
        if specification is None or specification.loader is None:
            raise ImportError(f"could not load synthetic test module: {test_path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        suite.addTests(loader.loadTestsFromModule(module))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    print(output, end="", flush=True)
    payload = {
        "passed": result.wasSuccessful(),
        "total_tests": int(result.testsRun),
        "passed_tests": int(result.testsRun - len(result.failures) - len(result.errors)),
        "failures": [f"{test.id()}: {detail}" for test, detail in result.failures],
        "errors": [f"{test.id()}: {detail}" for test, detail in result.errors],
        "skipped": [f"{test.id()}: {reason}" for test, reason in result.skipped],
        "modules": list(selected),
        "excluded_module": "test_stage2_dataset.py",
        "excluded_reason": "contains a real ASAP smoke test that opens the test split",
        "asap_test_midi_accessed": False,
    }
    destination = Path(output_dir) / "tests" / "test_results.json"
    atomic_json(destination, payload)
    if not result.wasSuccessful():
        raise RuntimeError(
            f"selected synthetic tests failed: failures={len(result.failures)} "
            f"errors={len(result.errors)}"
        )
    return payload


def _split_metadata(split_csv: Path) -> dict[str, Any]:
    with split_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row.get("split") == split]
        result[f"{split}_performance_count"] = len(selected)
        result[f"{split}_piece_count"] = len({row.get("piece_id") for row in selected})
    result.update(
        split_csv=str(split_csv),
        split_csv_sha256=file_sha256(split_csv),
        pipeline_splits=["train", "validation"],
        test_rows_passed_to_pipeline=0,
    )
    return result


def _runtime_metadata(environ: Mapping[str, str]) -> dict[str, Any]:
    warnings: list[str] = []
    gpu: dict[str, Any] = {
        "cuda_available": False,
        "visible_device_count": 0,
        "name": "unknown",
        "uuid": "unknown",
    }
    torch_version = cuda_version = "unknown"
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "unknown"
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["visible_device_count"] = int(torch.cuda.device_count())
        if gpu["visible_device_count"]:
            gpu["name"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError) as exc:
        warnings.append(f"torch runtime metadata: {type(exc).__name__}: {exc}")
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
        if query.returncode == 0 and len(rows) == 1:
            gpu["uuid"], gpu["nvidia_smi_name"] = [part.strip() for part in rows[0].split(",", 1)]
        else:
            warnings.append("nvidia-smi did not report exactly one visible GPU")
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        warnings.append(f"nvidia-smi metadata: {type(exc).__name__}: {exc}")

    try:
        current_user = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        current_user = "unknown"
    host_user, _ = _environment_value(
        environ, ("STAGE2_HOST_USERNAME", "HOST_USERNAME", "HOST_USER")
    )
    host_uid, _ = _environment_value(environ, ("STAGE2_HOST_UID", "HOST_UID"))
    return {
        "pytorch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu": gpu,
        "container_name": environ.get("STAGE2_CONTAINER_NAME", "ilkyun-marg-pedaling-dev"),
        "container_hostname": socket.gethostname(),
        "container_visible_cuda_index": 0,
        "cuda_visible_devices": environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "host_physical_gpu_mapping": environ.get("STAGE2_HOST_GPU_MAPPING", "unknown"),
        "host_username": host_user or current_user,
        "host_uid": int(host_uid) if host_uid and host_uid.isdigit() else "unknown",
        "container_username": current_user,
        "container_uid": os.getuid(),
        "host_repository_path": environ.get(
            "STAGE2_HOST_REPOSITORY_PATH",
            "/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling",
        ),
        "container_repository_path": str(PROJECT_ROOT),
        "warnings": warnings,
    }


def enrich_configuration(
    configuration: MutableMapping[str, Any],
    *,
    output_dir: Path,
    run_id: str,
    status: Mapping[str, Any],
    git_metadata: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    split_metadata: Mapping[str, Any],
    existing_run_action: Mapping[str, Any],
) -> dict[str, Any]:
    configuration.update(
        {
            "run_id": run_id,
            "launch_utc_timestamp": status["start_time_utc"],
            "launch_kst_timestamp": status["start_time_kst"],
            "output_dir": str(output_dir),
            "class_names": list(CLASS_NAMES),
            "class_boundaries": [list(bounds) for bounds in CLASS_BOUNDS],
            "representatives": REPRESENTATIVES.tolist(),
            "representative_policy": "canonical_interval_midpoint",
            "architecture": "official PT encoder + four independent Linear(768,5) heads",
            "loss": "standard_unweighted_5class_cross_entropy",
            "overlap_decoding": "average raw logits -> one argmax -> canonical representative",
            "dataset_class": "Stage2PedalDataset",
            "tokenizer": "pinned PT midi_to_ids/_PinnedTokenizerConfig",
            "input_grammar": [
                "Pitch", "IOI", "Velocity", "Duration",
                "MASK", "MASK", "MASK", "MASK",
            ],
            "cache_mode": "preload",
            "scheduler": None,
            "split_csv_sha256": split_metadata["split_csv_sha256"],
            "train_performance_count": split_metadata["train_performance_count"],
            "validation_performance_count": split_metadata["validation_performance_count"],
            "train_piece_count": split_metadata["train_piece_count"],
            "validation_piece_count": split_metadata["validation_piece_count"],
            "test_rows_passed_to_pipeline": 0,
            "project_git_commit": git_metadata["commit"],
            "project_git_dirty_status": git_metadata["dirty_status"],
            "project_git_dirty": git_metadata["dirty"],
            "git_metadata": dict(git_metadata),
            "runtime_metadata": dict(runtime_metadata),
            "gpu_name": runtime_metadata["gpu"]["name"],
            "gpu_uuid": runtime_metadata["gpu"]["uuid"],
            "tmux_session_name": status["tmux_session"],
            "host_pid": status["host_pid"],
            "container_pid": status["container_pid"],
            "existing_run_action": dict(existing_run_action),
            "asap_test_midi_accessed": False,
        }
    )
    return dict(configuration)


def _existing_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / STATUS_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _meaningful_existing_entries(output_dir: Path) -> set[str]:
    if not output_dir.is_dir():
        return set()
    return {
        path.name
        for path in output_dir.iterdir()
        if path.name != LOCK_NAME and not path.name.startswith(f"{LOCK_NAME}.")
    }


def _archive_incomplete_output(output_dir: Path, run_id: str) -> Path:
    if output_dir == output_dir.parent or output_dir.name in {"", ".", ".."}:
        raise ValueError(f"refusing to archive broad output path: {output_dir}")
    archive = output_dir.parent / (
        f"{output_dir.name}.failed.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f".{run_id[:12]}"
    )
    if archive.exists():
        archive = archive.with_name(f"{archive.name}.{time.time_ns()}")
    os.replace(output_dir, archive)
    _fsync_directory(output_dir.parent)
    return archive


def _format_json(payload: Any) -> str:
    return "```json\n" + json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n```"


def _comparison_row(evaluation: Mapping[str, Any], model: str) -> dict[str, Any]:
    comparison = evaluation.get("comparison", {})
    if isinstance(comparison, Mapping):
        row = comparison.get(model, {})
        return dict(row) if isinstance(row, Mapping) else {}
    return {}


def _interpretation(evaluation: Mapping[str, Any]) -> list[str]:
    row = _comparison_row(evaluation, "endpoint_aware_5class_argmax")
    reference = _comparison_row(evaluation, "ce_128_posterior_median")
    statements: list[str] = []
    recalls = {name: row.get(f"{name}_recall") for name in ("zero", "low", "mid", "high", "full")}
    zero_recall = [name.upper() for name, value in recalls.items() if value == 0]
    if zero_recall:
        statements.append(f"Class collapse diagnostic: zero recall for {', '.join(zero_recall)}.")
    else:
        statements.append("All-class recall diagnostic: no class has exactly zero recall." if recalls and all(value is not None for value in recalls.values()) else "Per-class recall diagnostic: 미확인.")
    low_recall = row.get("low_recall")
    if isinstance(low_recall, (int, float)):
        statements.append(f"ZERO/LOW separation: LOW recall={low_recall:.6f}; detailed directional confusion is reported above.")
    state_accuracy = row.get("off_on_state_accuracy")
    depth_accuracy = row.get("on_region_3way_accuracy")
    if isinstance(state_accuracy, (int, float)) and isinstance(depth_accuracy, (int, float)):
        statements.append(
            f"Renderer state versus depth localization: OFF/ON accuracy={state_accuracy:.6f}, "
            f"ON-region 3-way accuracy={depth_accuracy:.6f}."
        )
    transition = row.get("binary_transition_f1")
    reference_transition = reference.get("binary_transition_f1")
    if isinstance(transition, (int, float)):
        suffix = f" versus reference {reference_transition:.6f}" if isinstance(reference_transition, (int, float)) else ""
        statements.append(f"Temporal transition diagnostic: binary transition F1={transition:.6f}{suffix}.")
    oracle = row.get("canonical_oracle_overall_mae")
    model_mae = row.get("canonical_decoded_overall_mae")
    gap = row.get("model_minus_oracle_overall_mae")
    if all(isinstance(value, (int, float)) for value in (oracle, model_mae, gap)):
        statements.append(
            f"Quantization/model decomposition: oracle MAE={oracle:.6f}, model MAE={model_mae:.6f}, gap={gap:.6f}."
        )
    assessment = evaluation.get("success_assessment")
    if isinstance(assessment, Mapping):
        statements.append(
            f"Predeclared decision: Criterion A={assessment.get('criterion_a')}, "
            f"Criterion B={assessment.get('criterion_b')}, safeguards={assessment.get('all_safeguards_passed')}, "
            f"final pass={assessment.get('passed')}."
        )
    return statements or ["Scientific interpretation metrics were not returned; 미확인."]


def render_training_report(
    configuration: Mapping[str, Any],
    test_results: Mapping[str, Any],
    train_oracle: Mapping[str, Any],
    overfit: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    """Render all 26 requested sections without inventing missing metrics."""

    five = _comparison_row(evaluation, "endpoint_aware_5class_argmax")
    argmax = _comparison_row(evaluation, "ce_128_argmax")
    primary = _comparison_row(evaluation, "ce_128_posterior_median")
    ordinal = _comparison_row(evaluation, "ordinal_lambda1_posterior_median")
    coarse = _comparison_row(evaluation, "coarse_to_fine_v0")
    new_model = "endpoint_aware_5class_argmax"

    def model_detail(group: str) -> dict[str, Any]:
        models = evaluation.get(group, {})
        if not isinstance(models, Mapping):
            return {}
        detail = models.get(new_model, {})
        return dict(detail) if isinstance(detail, Mapping) else {}

    classification_detail = model_detail("classification_metrics")
    renderer_detail = model_detail("renderer_aware_metrics")
    decoded_detail = model_detail("decoded_value_metrics")
    confusion_detail = {
        "confusion_matrix": classification_detail.get("confusion_matrix"),
        **{
            key: value
            for key, value in classification_detail.items()
            if "_to_" in key or "collapse" in key
        },
    }
    renderer_state_detail = {
        key: renderer_detail[key]
        for key in (
            "state", "zero_vs_nonzero", "on_region_depth_class_accuracy",
            "on_region_macro_f1", "on_region_per_class_f1",
        )
        if key in renderer_detail
    }
    renderer_transition_detail = {
        key: renderer_detail[key]
        for key in (
            "off_to_on_transition", "on_to_off_transition",
            "combined_binary_transition", "steady_position_state_count",
            "steady_position_state_accuracy", "transition_direction_accuracy",
            "transition_timing_sample_offset", "short_repedal_like",
        )
        if key in renderer_detail
    }
    classification_keys = (
        "five_class_token_accuracy", "five_class_exact_note_accuracy", "five_class_macro_f1",
        "five_class_weighted_f1", "five_class_micro_f1", "predicted_zero_ratio",
        "predicted_low_ratio", "predicted_mid_ratio", "predicted_high_ratio", "predicted_full_ratio",
    )
    class_keys = tuple(
        f"{name}_{metric}"
        for name in ("zero", "low", "mid", "high", "full")
        for metric in ("precision", "recall", "f1", "support")
    )
    renderer_keys = tuple(
        key for key in five if any(token in key for token in ("off_on", "balanced_accuracy", "zero_vs_nonzero", "on_region"))
    )
    transition_keys = tuple(
        key for key in five if any(token in key for token in ("transition", "repedal", "steady", "delta_mae"))
    )
    decoded_keys = tuple(
        key for key in five if any(token in key for token in ("canonical_decoded", "tolerance", "maximum", "quantile"))
    )
    confusion = {key: value for key, value in five.items() if "confusion" in key or "collapse" in key}
    assessment = evaluation.get("success_assessment", {"status": "미확인"})
    unweighted_baseline: dict[str, Any] = {}
    if configuration.get("loss_configuration", {}).get("name") == "class_weighted_cross_entropy_inverse_sqrt_train_global":
        baseline_csv = PROJECT_ROOT / "analysis" / "stage2_encoder_only_5class_v0" / "validation" / "validation_comparison.csv"
        if baseline_csv.is_file():
            with baseline_csv.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("model") == "endpoint_aware_5class_argmax":
                        unweighted_baseline = dict(row)
                        break
    report = f"""# Stage 2 Encoder-only Endpoint-aware 5-class Pedal Training v0

## 1. 연구 질문과 가설

동일한 official pretrained PT encoder, ASAP piece-wise split, optimizer, seed, windowing, and training budget에서 Pedal1–4 output만 five coarse classes로 바꾸었을 때 128-class posterior fragmentation과 endpoint collapse가 완화되는지를 validation-only로 검증했다.

## 2. 5-class 정의

| ID | Class | Raw target range |
| ---: | --- | --- |
{os.linesep.join(f'| {index} | {name} | {low}–{high} |' for index, (name, (low, high)) in enumerate(zip(CLASS_NAMES, CLASS_BOUNDS)))}

## 3. Canonical representative 정의

고정 representative는 `{REPRESENTATIVES.tolist()}`이며 policy는 `canonical_interval_midpoint`이다. Train/validation 분포로 재추정하지 않았다.

## 4. LOW를 1–63으로 합친 이유

CC64 sustain threshold 미만의 nonzero physical pedal positions를 하나의 OFF-region depth class로 표현하여 flat 128-way fragmentation을 줄이는 고정 가설이다.

## 5. ZERO와 LOW를 분리한 이유

완전 release endpoint 0과 subthreshold but physically nonzero 1–63을 구분하여 release collapse를 직접 측정한다.

## 6. 데이터 및 piece-wise split

{_format_json({key: configuration.get(key, '미확인') for key in ('asap_root','split_csv','split_csv_sha256','train_performance_count','train_piece_count','validation_performance_count','validation_piece_count','window_notes','stride_notes','test_rows_passed_to_pipeline')})}

## 7. 모델 architecture

{configuration.get('architecture', '미확인')}. Encoder는 unfrozen이고 four prediction heads는 independent Linear(768,5)이다.

## 8. Loss와 decoding

Loss configuration은 아래와 같으며, weighted run에서는 ASAP train Pedal1–4 global weight만 사용한다. Final reconstruction은 raw logits window-average → one argmax → canonical representative 순서이며 smoothing/calibration은 없다.

{_format_json(configuration.get('loss_configuration', {'status': '미확인'}))}

## 9. Implementation tests

{_format_json(test_results)}

## 10. Pedal-rich GPU overfit result

{_format_json(overfit)}

## 11. Training summary

{_format_json(training)}

## 12. Best epoch와 validation loss

- Best epoch: `{training.get('best_epoch', '미확인')}`
- Best validation five-class CE: `{training.get('best_validation_loss', '미확인')}`
- Stop reason: `{training.get('stop_reason', '미확인')}`

## 13. 5-class classification metrics

{_format_json({key: five.get(key, '미확인') for key in classification_keys})}

## 14. Class별 precision/recall/F1

{_format_json(classification_detail or {key: five.get(key, '미확인') for key in class_keys})}

## 15. ZERO/LOW confusion analysis

{_format_json(confusion_detail if classification_detail else (confusion or {'status': '미확인'}))}

## 16. OFF/ON renderer-aware metrics

{_format_json(renderer_state_detail or ({key: five[key] for key in renderer_keys} if renderer_keys else {'status': '미확인'}))}

## 17. Transition 및 repedal analysis

{_format_json(renderer_transition_detail or ({key: five[key] for key in transition_keys} if transition_keys else {'status': '미확인'}))}

Short repedal-like metric은 `>=64 → <64 → >=64`, low run ≤4 flattened Pedal samples인 sample-based diagnostic이며 millisecond timing metric이 아니다.

## 18. Canonical decoded-value metrics

{_format_json(decoded_detail or ({key: five[key] for key in decoded_keys} if decoded_keys else {'status': '미확인'}))}

## 19. Validation quantization oracle와 model gap

{_format_json(evaluation.get('validation_quantization_oracle', {'status': '미확인'}))}

Model row oracle/gap:

{_format_json({key: value for key, value in five.items() if 'oracle' in key or 'model_minus' in key})}

Train canonical oracle:

{_format_json(train_oracle)}

## 20. 기존 CE argmax 및 posterior-median rebinned comparison

Baseline CE argmax:

{_format_json(argmax or {'status': '미확인'})}

Primary reference:

{_format_json(primary or {'status': '미확인'})}

New five-class:

{_format_json(five or {'status': '미확인'})}

### Existing unweighted five-class baseline comparison

기존 baseline checkpoint를 재학습하지 않고, 동일한 validation aggregate CSV의 `endpoint_aware_5class_argmax` row를 참조했다.

Existing unweighted five-class:

{_format_json(unweighted_baseline or {'status': 'baseline CSV row not found'})}

Current weighted five-class:

{_format_json(five or {'status': '미확인'})}

## 21. Ordinal/coarse-to-fine secondary comparison

Ordinal lambda 1.0 posterior median:

{_format_json(ordinal or {'status': '미확인'})}

Coarse-to-fine v0:

{_format_json(coarse or {'status': '미확인'})}

## 22. Success criteria 판정

{_format_json(assessment)}

## 23. 과학적 해석

{os.linesep.join(f'- {statement}' for statement in _interpretation(evaluation))}

## 24. 한계

- Model selection과 결과 해석은 ASAP validation에 한정된다.
- LOW 내부 raw-value error는 OFF-state semantics와 quantization error를 함께 반영한다.
- Four pedal slots are conditionally predicted from encoder states without explicit previous-pedal conditioning or smoothing.
- Short repedal은 variable-duration IOI를 가진 sample-count diagnostic이다.

## 25. 권장 다음 단계

{'모든 predeclared criterion/safeguard를 통과했으므로 이 checkpoint를 controlled five-class candidate로 동결하고 사용자 검토 후 다음 결정을 내린다.' if isinstance(assessment, Mapping) and assessment.get('passed') is True else '사전 기준을 통과하지 못했으므로 자동으로 test/MAESTRO/추가 architecture 실험을 진행하지 않고 측정된 failure mode를 먼저 검토한다.'}

## 26. ASAP test split 미사용 확인

- Evaluation split: `{evaluation.get('split', '미확인')}`
- Evaluation test rows used: `{evaluation.get('test_rows_used', '미확인')}`
- Training/evaluation pipeline test rows: `{configuration.get('test_rows_passed_to_pipeline', '미확인')}`
- Selected unit tests excluded `test_stage2_dataset.py`, whose real-data smoke includes ASAP test MIDI.
- This run generated no ASAP test metric, prediction, representative, checkpoint selection, MIDI, or report input.
"""
    return report.rstrip() + "\n"


def _choose_run_id(output_dir: Path) -> tuple[str, dict[str, Any] | None]:
    existing = _existing_status(output_dir)
    existing_id = existing.get("run_id") if existing else None
    run_id = str(existing_id) if existing_id else (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    )
    return run_id, existing


def _assert_gpu_runtime(runtime: Mapping[str, Any], configuration: Mapping[str, Any]) -> None:
    gpu = runtime["gpu"]
    if not gpu.get("cuda_available") or int(gpu.get("visible_device_count", 0)) != 1:
        raise RuntimeError(f"full five-class run requires exactly one visible CUDA GPU: {gpu}")
    expected = configuration.get("expected_gpu_uuid")
    observed = gpu.get("uuid")
    if expected and expected != "unknown" and observed != expected:
        raise RuntimeError(f"GPU UUID mismatch: expected {expected}, observed {observed}")


def run(output_dir: str | Path) -> int:
    """Execute the complete controlled experiment and validation-only report."""

    from .evaluate_five_class import evaluate
    from .train_five_class import (
        canonical_train_oracle,
        default_configuration,
        pedal_rich_overfit,
        train,
    )

    output = Path(output_dir).resolve()
    report_name = os.environ.get("STAGE2_REPORT_NAME", REPORT_NAME)
    overrides_text = os.environ.get("STAGE2_CONFIGURATION_OVERRIDES_JSON")
    configuration_overrides = json.loads(overrides_text) if overrides_text else {}
    if not isinstance(configuration_overrides, Mapping):
        raise ValueError("STAGE2_CONFIGURATION_OVERRIDES_JSON must be a JSON object")
    output.parent.mkdir(parents=True, exist_ok=True)
    host_pid = _parse_optional_pid(
        os.environ.get("STAGE2_HOST_PID") or os.environ.get("HOST_PID")
    )
    tmux_session = os.environ.get("STAGE2_TMUX_SESSION", DEFAULT_TMUX_SESSION)
    run_id, previous_status = _choose_run_id(output)
    lock = acquire_run_lock(output, run_id, host_pid=host_pid, container_pid=os.getpid())
    existing_action: dict[str, Any] = {
        "action": "new",
        "stale_lock_archives": list(lock.stale_archives),
        "archive_path": None,
    }

    # A complete run is immutable. An interrupted run resumes only with an
    # epoch-boundary last.pt; otherwise preserve the whole directory as failed.
    if previous_status and (previous_status.get("completed") or previous_status.get("stage") == "complete"):
        release_run_lock(lock, "refused_completed")
        raise FileExistsError(f"completed five-class output is immutable: {output}")
    resume_path: Path | None = None
    entries = _meaningful_existing_entries(output)
    if entries and previous_status:
        candidate = output / "last.pt"
        config_path = output / "config.json"
        if candidate.is_file() and candidate.stat().st_size > 0 and config_path.is_file():
            resume_path = candidate
            existing_action["action"] = "resume_epoch_boundary_candidate"
        else:
            release_run_lock(lock, "archive_incomplete")
            archive = _archive_incomplete_output(output, run_id)
            existing_action.update(action="archive_unresumable", archive_path=str(archive))
            run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
            previous_status = None
            lock = acquire_run_lock(output, run_id, host_pid=host_pid, container_pid=os.getpid())
    elif entries and not previous_status:
        release_run_lock(lock, "archive_unknown_incomplete")
        archive = _archive_incomplete_output(output, run_id)
        existing_action.update(action="archive_unknown_unresumable", archive_path=str(archive))
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
        lock = acquire_run_lock(output, run_id, host_pid=host_pid, container_pid=os.getpid())

    output.mkdir(parents=True, exist_ok=True)
    (output / "tests").mkdir(exist_ok=True)
    status_payload = initial_run_status(
        run_id,
        host_pid,
        os.getpid(),
        tmux_session,
        start_time_utc=(previous_status or {}).get("start_time_utc"),
        start_time_kst=(previous_status or {}).get("start_time_kst"),
    )
    status_payload.update(
        resumed=resume_path is not None,
        resume_checkpoint=str(resume_path) if resume_path else None,
        existing_run_action=existing_action,
        lock_path=str(lock.path),
    )
    status = AtomicRunStatus(output / STATUS_NAME, status_payload)
    log_path = output / "train.log"
    outcome = "failed"
    failure_stage = "initializing"
    configuration: dict[str, Any] = {}
    try:
        with log_path.open("a", encoding="utf-8") as train_log:
            tee_out, tee_err = Tee(sys.stdout, train_log), Tee(sys.stderr, train_log)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                print(f"FIVE CLASS RUN START run_id={run_id} output={output}", flush=True)
                print(f"lock={lock.payload} existing_action={existing_action}", flush=True)

                failure_stage = "initializing"
                base = default_configuration(str(output))
                if resume_path:
                    saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
                    configuration = dict(saved)
                    for key, value in base.items():
                        configuration.setdefault(key, value)
                    configuration["resume"] = str(resume_path)
                else:
                    configuration = dict(base)
                    configuration.update(configuration_overrides)
                split_path = Path(configuration.get("split_csv", DEFAULT_SPLIT_CSV)).resolve()
                split = _split_metadata(split_path)
                git = collect_git_metadata(PROJECT_ROOT)
                runtime = _runtime_metadata(os.environ)
                configuration = enrich_configuration(
                    configuration,
                    output_dir=output,
                    run_id=run_id,
                    status=status.payload,
                    git_metadata=git,
                    runtime_metadata=runtime,
                    split_metadata=split,
                    existing_run_action=existing_action,
                )
                _assert_gpu_runtime(runtime, configuration)
                status.update(gpu_uuid=runtime["gpu"]["uuid"])
                atomic_json(output / "config.json", configuration)
                for warning in (*git.get("warnings", []), *runtime.get("warnings", [])):
                    print(f"WARNING metadata: {warning}", flush=True)

                failure_stage = "testing"
                status.update(stage="testing")
                tests = run_selected_synthetic_tests(PROJECT_ROOT, output)
                train_oracle = canonical_train_oracle(output)
                if (
                    train_oracle.get("source_split") != "train"
                    or train_oracle.get("test_split_accessed") is not False
                ):
                    raise RuntimeError(
                        "canonical oracle did not prove train-only provenance"
                    )

                failure_stage = "overfit"
                status.update(stage="overfit")
                overfit = pedal_rich_overfit(output)
                if (
                    not overfit.get("passed", False)
                    or overfit.get("source_split") != "train"
                    or overfit.get("test_split_accessed") is not False
                ):
                    raise RuntimeError(
                        "pedal-rich overfit gate failed train-only provenance"
                    )

                failure_stage = "training"
                status.update(stage="training")

                def epoch_callback(row: Mapping[str, Any]) -> None:
                    status.update(
                        stage="training",
                        current_epoch=int(row.get("epoch", status.payload["current_epoch"])),
                        current_step=int(row.get("global_optimizer_step", status.payload["current_step"])),
                        best_epoch=int(row.get("best_epoch", status.payload["best_epoch"])),
                        best_validation_loss=row.get(
                            "best_validation_loss", status.payload["best_validation_loss"]
                        ),
                        patience_counter=row.get("early_stopping_counter"),
                    )

                def progress_callback(update: Mapping[str, Any]) -> None:
                    progress_stage = str(update.get("stage", "training"))
                    mapped_stage = "preloading" if progress_stage == "preloading" else "training"
                    status.update(
                        stage=mapped_stage,
                        current_epoch=int(update.get("epoch", status.payload["current_epoch"])),
                        current_step=int(update.get("global_step", status.payload["current_step"])),
                        current_batch=update.get("batch"),
                        total_batches=update.get("batches"),
                        latest_finite_loss=update.get("loss"),
                        preload_split=update.get("split"),
                    )

                training_start_time_utc = utc_now()
                training_start_time_kst = kst_now()
                status.update(
                    stage="training",
                    training_start_time_utc=training_start_time_utc,
                    training_start_time_kst=training_start_time_kst,
                )
                training = train(
                    configuration,
                    epoch_callback=epoch_callback,
                    progress_callback=progress_callback,
                )
                configuration = json.loads((output / "config.json").read_text(encoding="utf-8"))
                training_end_time_utc = utc_now()
                training_end_time_kst = kst_now()
                training = dict(training)
                training.update(
                    training_start_time_utc=training_start_time_utc,
                    training_start_time_kst=training_start_time_kst,
                    training_end_time_utc=training_end_time_utc,
                    training_end_time_kst=training_end_time_kst,
                )
                status.update(
                    stage="validation",
                    current_epoch=int(training.get("completed_epoch", status.payload["current_epoch"])),
                    current_step=int(training.get("global_optimizer_step", status.payload["current_step"])),
                    best_epoch=int(training.get("best_epoch", status.payload["best_epoch"])),
                    best_validation_loss=training.get(
                        "best_validation_loss", status.payload["best_validation_loss"]
                    ),
                    training_start_time_utc=training_start_time_utc,
                    training_start_time_kst=training_start_time_kst,
                    training_end_time_utc=training_end_time_utc,
                    training_end_time_kst=training_end_time_kst,
                    early_stopped=training.get("early_stopped"),
                )

                failure_stage = "validation"
                evaluation = evaluate(output)
                if evaluation.get("split") != "validation" or evaluation.get("test_rows_used") != 0:
                    raise RuntimeError("evaluator did not prove validation-only scope")

                failure_stage = "comparison"
                status.update(stage="comparison")
                comparison = evaluation.get("comparison")
                expected_models = {
                    "ce_128_argmax",
                    "ce_128_posterior_median",
                    "ordinal_lambda1_posterior_median",
                    "coarse_to_fine_v0",
                    "endpoint_aware_5class_argmax",
                }
                if not isinstance(comparison, Mapping) or not expected_models.issubset(comparison):
                    raise RuntimeError("validation comparison is missing required model rows")

                failure_stage = "reporting"
                status.update(stage="reporting")
                report = render_training_report(
                    configuration, tests, train_oracle, overfit, training, evaluation
                )
                atomic_text(output / report_name, report)
                required = [
                    output / "config.json", output / STATUS_NAME, log_path,
                    output / "metrics.csv", output / "last.pt", output / "best.pt",
                    output / "tests" / "test_results.json",
                    output / "tests" / "pedal_rich_overfit.json",
                    output / "tests" / "canonical_oracle_train.json",
                    output / "validation" / "validation_comparison.csv",
                    output / "validation" / "per_performance_metrics.csv",
                    output / "validation" / "per_slot_metrics.csv",
                    output / "validation" / "five_class_confusion.csv",
                    output / "validation" / "per_class_metrics.csv",
                    output / "validation" / "class_distributions.csv",
                    output / "validation" / "renderer_aware_metrics.json",
                    output / "validation" / "canonical_quantization_gap.json",
                    output / "validation" / "error_quantiles.json",
                    output / "validation" / "reference_rebinned_metrics.csv",
                    output / report_name,
                ]
                missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"required final artifacts missing or empty: {missing}")
                assessment = evaluation.get("success_assessment", {})
                status.update(
                    stage="complete",
                    completed=True,
                    failed=False,
                    failure_stage=None,
                    exception_type=None,
                    exception_message=None,
                    end_time_utc=utc_now(),
                    end_time_kst=kst_now(),
                    final_pass=assessment.get("passed") if isinstance(assessment, Mapping) else None,
                    output_files=evaluation.get("output_files"),
                    asap_test_midi_accessed=False,
                )
                print(
                    f"FIVE CLASS RUN COMPLETE best_epoch={training.get('best_epoch')} "
                    f"best_validation_loss={training.get('best_validation_loss')} "
                    f"pass={status.payload.get('final_pass')}",
                    flush=True,
                )
                outcome = "complete"
                return 0
    except BaseException as exc:
        resume_candidate = output / "last.pt"
        try:
            status.fail(
                failure_stage,
                exc,
                end_time_utc=utc_now(),
                end_time_kst=kst_now(),
                last_checkpoint_exists=resume_candidate.is_file(),
                best_checkpoint_exists=(output / "best.pt").is_file(),
                exact_epoch_resume_candidate=(
                    str(resume_candidate)
                    if resume_candidate.is_file() and resume_candidate.stat().st_size > 0
                    else None
                ),
                asap_test_midi_accessed=False,
            )
        except BaseException as status_exc:
            print(
                f"FATAL: could not write failure status: {type(status_exc).__name__}: {status_exc}",
                file=sys.stderr,
                flush=True,
            )
        failure_summary = (
            f"FIVE CLASS RUN FAILED stage={failure_stage}: "
            f"{type(exc).__name__}: {exc}"
        )
        failure_traceback = traceback.format_exc()
        try:
            with log_path.open("a", encoding="utf-8") as failure_log:
                failure_log.write(f"{failure_summary}\n{failure_traceback}")
                failure_log.flush()
                os.fsync(failure_log.fileno())
        except OSError as log_exc:
            print(
                f"FATAL: could not append failure to train.log: {log_exc}",
                file=sys.stderr,
                flush=True,
            )
        print(
            failure_summary,
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        archived_lock = release_run_lock(lock, outcome)
        if archived_lock is not None:
            print(f"run lock archived: {archived_lock}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default=None)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.report_name:
        os.environ["STAGE2_REPORT_NAME"] = arguments.report_name
    return run(arguments.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
