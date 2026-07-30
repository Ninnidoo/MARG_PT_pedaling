from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SESSION_TEMPLATE_HEADINGS = [
    "오늘의 목표",
    "완료한 작업",
    "검증된 결과",
    "주요 결정",
    "발생한 문제와 해결",
    "미해결 문제",
    "다음 세션 우선순위",
    "다음 ChatGPT 대화용 Follow-up Prompt",
]

DOC_PATHS = {
    "experiment_log": Path("docs/EXPERIMENT_LOG.md"),
    "decisions": Path("docs/DECISIONS.md"),
    "next_session": Path("docs/NEXT_SESSION.md"),
    "troubleshooting": Path("docs/TROUBLESHOOTING.md"),
}

METADATA_SEARCH_ROOTS = [Path("experiments"), Path("outputs"), Path("docs/session_logs")]
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", "third_party", "checkpoints"}
METADATA_MAX_BYTES = 256 * 1024
METADATA_LIMIT = 5


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def run_git(root: Path, args: list[str]) -> tuple[bool, str]:
    cmd = ["git", "-c", f"safe.directory={root.as_posix()}", *args]
    try:
        result = subprocess.run(
            cmd,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"미확인: git {' '.join(args)} 실행 실패: {exc}"

    output = result.stdout.strip()
    if result.returncode != 0:
        error_output = f"{output}\n{result.stderr.strip()}".strip()
        return False, f"미확인: git {' '.join(args)} 실패: {error_output}"
    return True, output


def slugify(title: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", title.strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-._")
    return normalized or "session"


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def lines_for_date(text: str, date_text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if date_text in line:
            lines.append(line)
    return dedupe(lines)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    target = heading.strip().casefold()
    collecting = False
    level = 0
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_level = len(stripped) - len(stripped.lstrip("#"))
            current_heading = stripped.lstrip("#").strip().casefold()
            if collecting and current_level <= level:
                break
            if current_heading == target:
                collecting = True
                level = current_level
                continue
        if collecting:
            out.append(line)
    return "\n".join(out).strip()


def extract_bullets(section: str) -> list[str]:
    bullets: list[str] = []
    for raw in section.splitlines():
        stripped = raw.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
        else:
            match = re.match(r"^\d+[.)]\s+(.*)$", stripped)
            if match:
                bullets.append(match.group(1).strip())
    return dedupe([item for item in bullets if item])


def format_bullets(items: list[str], fallback: str = "미확인") -> str:
    if not items:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in items)


def is_metadata_candidate(path: Path) -> bool:
    name = path.name.lower()
    if not path.is_file():
        return False
    if path.suffix.lower() != ".json":
        return False
    if "metadata" not in name:
        return False
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return False
    try:
        return path.stat().st_size <= METADATA_MAX_BYTES
    except OSError:
        return False


def recent_metadata_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for rel_root in METADATA_SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            if is_metadata_candidate(path):
                candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:METADATA_LIMIT]


def find_value(data: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            found = find_value(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_value(value, keys)
            if found is not None:
                return found
    return None


def summarize_metadata(path: Path, root: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{relative_display(path, root)}: 미확인: metadata 읽기 실패: {exc}"]

    field_map: list[tuple[str, tuple[str, ...]]] = [
        ("date", ("date", "run_date")),
        ("model", ("model", "model_name")),
        ("checkpoint", ("checkpoint", "checkpoint_repo_id", "checkpoint_id")),
        ("repository_commit", ("repository_commit", "commit", "model_commit")),
        ("input", ("input", "input_midi", "input_path")),
        ("output", ("output", "output_midi", "output_path")),
        ("seed", ("seed", "random_seed")),
        ("temperature", ("temperature",)),
        ("top_p", ("top_p", "top_p_value")),
        ("device", ("device",)),
        ("dtype", ("dtype", "torch_dtype")),
        ("inference_seconds", ("inference_seconds", "inference_time", "runtime_seconds")),
        ("generated_tokens", ("generated_tokens", "generated_token_count", "token_count")),
        ("note_count", ("note_count", "notes")),
        ("cc64_event_count", ("cc64_event_count", "pedal_event_count")),
    ]

    parts = [relative_display(path, root)]
    for label, keys in field_map:
        value = find_value(data, keys)
        if value is not None:
            parts.append(f"{label}={value}")
    return ["; ".join(parts)]


def build_summary(root: Path, title: str, date_text: str) -> str:
    docs = {name: read_text_if_exists(root / rel_path) for name, rel_path in DOC_PATHS.items()}

    ok_status, git_status = run_git(root, ["status", "--short"])
    ok_diff, git_diff = run_git(root, ["diff", "--name-status"])
    ok_untracked, git_untracked = run_git(root, ["ls-files", "--others", "--exclude-standard"])

    experiment_lines = lines_for_date(docs["experiment_log"], date_text)
    decision_lines = lines_for_date(docs["decisions"], date_text)
    verified_lines = [line for line in experiment_lines if re.search(r"success|verified|완료|검증", line, re.IGNORECASE)]

    next_topics = extract_bullets(extract_section(docs["next_session"], "Topics For The Next Research Session"))
    open_items = extract_bullets(extract_section(docs["troubleshooting"], "Still Open or Watch Items"))

    metadata_summaries: list[str] = []
    for metadata_path in recent_metadata_files(root):
        metadata_summaries.extend(summarize_metadata(metadata_path, root))

    git_items: list[str] = []
    if ok_status and git_status:
        git_items.append("현재 Git 변경 상태:")
        git_items.extend(f"  {line}" for line in git_status.splitlines())
    elif ok_status:
        git_items.append("현재 Git 변경 상태: clean")
    else:
        git_items.append(git_status)

    if ok_diff and git_diff:
        git_items.append("수정된 tracked 파일:")
        git_items.extend(f"  {line}" for line in git_diff.splitlines())
    if ok_untracked and git_untracked:
        git_items.append("untracked 파일:")
        git_items.extend(f"  {line}" for line in git_untracked.splitlines())

    follow_up_prompt = (
        "```text\n"
        f"MARG_research 프로젝트의 {date_text} 세션 로그를 바탕으로 다음 연구 세션을 설계해줘.\n"
        "우선 AGENTS.md, docs/EXPERIMENT_LOG.md, docs/DECISIONS.md, docs/NEXT_SESSION.md를 근거로 삼고, "
        "repedaling 정의나 CC64 차이에 대한 원인은 아직 확정하지 말아줘.\n"
        "다음 세션에서는 연구 질문 재정리, pedal-heavy score 선정 기준, baseline pedal behavior 분석 계획, "
        "human performance MIDI 데이터 확보 전략을 순서대로 점검하고 싶어.\n"
        "```"
    )

    sections: list[str] = [f"# {date_text} — {title}", ""]

    sections.extend([
        "## 오늘의 목표",
        format_bullets([
            f"세션 제목: {title}",
            "미확인: 세션 시작 시 사용자가 명시한 목표 문장은 자동 입력 문서에서 별도로 확인되지 않았습니다.",
        ]),
        "",
        "## 완료한 작업",
        format_bullets(experiment_lines, f"미확인: docs/EXPERIMENT_LOG.md에서 {date_text} 항목을 찾지 못했습니다."),
        "",
        "근거로 확인한 Git 상태:",
        format_bullets(git_items),
        "",
        "## 검증된 결과",
        format_bullets(verified_lines, "미확인: 오늘 날짜의 검증 완료 항목을 실험 로그에서 찾지 못했습니다."),
        "",
        "최근 실험 metadata:",
        format_bullets(metadata_summaries, "미확인: 최근 metadata JSON을 찾지 못했습니다."),
        "",
        "## 주요 결정",
        format_bullets(decision_lines, f"미확인: docs/DECISIONS.md에서 {date_text} 항목을 찾지 못했습니다."),
        "",
        "## 발생한 문제와 해결",
        format_bullets([], "미확인: 오늘 새로 발생하고 해결된 문제는 자동 입력 문서에서 확인되지 않았습니다."),
        "",
        "## 미해결 문제",
        format_bullets(open_items, "미확인: 미해결 항목을 찾지 못했습니다."),
        "",
        "## 다음 세션 우선순위",
        format_bullets(next_topics, "미확인: docs/NEXT_SESSION.md에서 다음 세션 항목을 찾지 못했습니다."),
        "",
        "## 다음 ChatGPT 대화용 Follow-up Prompt",
        follow_up_prompt,
        "",
    ])

    return "\n".join(sections)


def output_path_for(root: Path, date_text: str, title: str) -> Path:
    return root / "docs" / "session_logs" / f"{date_text}_{slugify(title)}.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a grounded local research session summary.")
    parser.add_argument("--title", required=True, help="Session title used in the H1 and output filename slug.")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Session date in YYYY-MM-DD format.")
    parser.add_argument("--dry-run", action="store_true", help="Print Markdown without writing a file.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing session log file.")
    return parser.parse_args()


def main() -> int:
    configure_output_encoding()
    args = parse_args()
    root = project_root()
    summary = build_summary(root, args.title, args.date)

    if args.dry_run:
        print(summary)
        return 0

    out_path = output_path_for(root, args.date, args.title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing session log: {out_path}", file=sys.stderr)
        print("Use a different --title or pass --overwrite.", file=sys.stderr)
        return 2
    out_path.write_text(summary, encoding="utf-8")
    print(f"Wrote session log: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



