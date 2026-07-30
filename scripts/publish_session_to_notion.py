from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - dry-run can still run without requests.
    requests = None  # type: ignore[assignment]

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"
RICH_TEXT_CHUNK = 1900
BLOCK_BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
PUBLISH_LOG_PATH = Path("docs/session_logs/.notion_publish_log.json")


class PublishError(RuntimeError):
    pass


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_root_env(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        raise PublishError("Missing .env in project root. Create it from .env.example before real upload.")

    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def require_env(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise PublishError(f"Missing {key} in project root .env")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_publish_log(root: Path) -> dict[str, Any]:
    path = root / PUBLISH_LOG_PATH
    if not path.exists():
        return {"version": 1, "entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublishError(f"Publish log is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PublishError(f"Publish log has invalid format: {path}")
    data.setdefault("version", 1)
    data.setdefault("entries", [])
    return data


def save_publish_log(root: Path, data: dict[str, Any]) -> None:
    path = root / PUBLISH_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_existing_entry(log: dict[str, Any], sha256: str) -> dict[str, Any] | None:
    entries = log.get("entries", [])
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("sha256") == sha256:
            return entry
    return None


def parse_markdown_title(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            remaining = "\n".join(lines[index + 1 :]).strip("\n")
            return title or "Untitled session", remaining
    return "Untitled session", markdown


def split_text(text: str, limit: int = RICH_TEXT_CHUNK) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + limit])
        start += limit
    return chunks


def rich_text(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": chunk}} for chunk in split_text(text)]


def text_block(block_type: str, text: str) -> dict[str, Any]:
    text = text if text else " "
    return {
        "object": "block",
        "type": block_type,
        block_type: {"rich_text": rich_text(text)},
    }


def code_blocks(text: str, language: str) -> list[dict[str, Any]]:
    notion_language = normalize_code_language(language)
    blocks = []
    for chunk in split_text(text, limit=RICH_TEXT_CHUNK):
        blocks.append(
            {
                "object": "block",
                "type": "code",
                "code": {"rich_text": rich_text(chunk), "language": notion_language},
            }
        )
    return blocks


def normalize_code_language(language: str) -> str:
    language = language.strip().lower()
    mapping = {
        "": "plain text",
        "text": "plain text",
        "txt": "plain text",
        "py": "python",
        "python": "python",
        "json": "json",
        "md": "markdown",
        "markdown": "markdown",
        "bash": "shell",
        "sh": "shell",
        "shell": "shell",
        "powershell": "powershell",
        "ps1": "powershell",
    }
    return mapping.get(language, "plain text")


def markdown_to_notion_blocks(markdown: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []
    in_code = False
    code_language = "plain text"
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            blocks.append(text_block("paragraph", " ".join(line.strip() for line in paragraph_lines).strip()))
            paragraph_lines = []

    for raw in markdown.splitlines():
        line = raw.rstrip("\n")
        fence = re.match(r"^```\s*([A-Za-z0-9_+.-]*)\s*$", line.strip())
        if fence:
            if in_code:
                blocks.extend(code_blocks("\n".join(code_lines), code_language))
                in_code = False
                code_language = "plain text"
                code_lines = []
            else:
                flush_paragraph()
                in_code = True
                code_language = fence.group(1) or "plain text"
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            block_type = {1: "heading_1", 2: "heading_2", 3: "heading_3"}[level]
            blocks.append(text_block(block_type, heading.group(2).strip()))
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            flush_paragraph()
            blocks.append(text_block("bulleted_list_item", bullet.group(1).strip()))
            continue

        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if numbered:
            flush_paragraph()
            blocks.append(text_block("numbered_list_item", numbered.group(1).strip()))
            continue

        paragraph_lines.append(line)

    if in_code:
        blocks.extend(code_blocks("\n".join(code_lines), code_language))
    flush_paragraph()
    return blocks


def batches(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def sanitize_message(message: str, token: str = "", parent_page_id: str = "") -> str:
    sanitized = message
    if token:
        sanitized = sanitized.replace(token, "[redacted-token]")
    if parent_page_id:
        sanitized = sanitized.replace(parent_page_id, "[redacted-parent-page-id]")
    return sanitized[:1000]


def notion_request(method: str, path: str, token: str, parent_page_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if requests is None:
        raise PublishError("Missing dependency: requests. Install with: python -m pip install -r requirements-notion.txt")

    url = f"{NOTION_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.request(method, url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout:
            last_error = f"HTTP timeout after {REQUEST_TIMEOUT_SECONDS}s"
        except requests.RequestException as exc:
            last_error = sanitize_message(str(exc), token, parent_page_id)
        else:
            if 200 <= response.status_code < 300:
                return response.json()

            retryable = response.status_code in {429, 500, 502, 503, 504, 529}
            try:
                error_data = response.json()
            except ValueError:
                error_data = {"message": response.text[:500]}
            code = error_data.get("code", "unknown_error")
            message = sanitize_message(str(error_data.get("message", "No error message")), token, parent_page_id)
            last_error = f"Notion API HTTP {response.status_code} code={code}: {message}"

            if not retryable or attempt == MAX_RETRIES:
                raise PublishError(last_error)

            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(int(retry_after))
            else:
                time.sleep(min(2 ** attempt, 8))
            continue

        if attempt < MAX_RETRIES:
            time.sleep(min(2 ** attempt, 8))

    raise PublishError(last_error or "Unknown Notion request failure")


def create_page(token: str, parent_page_id: str, title: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    title_text = title[:RICH_TEXT_CHUNK]
    first_batch = blocks[:BLOCK_BATCH_SIZE]
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {"title": rich_text(title_text)},
        "children": first_batch,
    }
    page = notion_request("POST", "/pages", token, parent_page_id, payload)
    page_id = page.get("id")
    if not page_id:
        raise PublishError("Notion create page response did not include a page id.")

    for batch in batches(blocks[BLOCK_BATCH_SIZE:], BLOCK_BATCH_SIZE):
        append_payload = {"children": batch}
        notion_request("PATCH", f"/blocks/{page_id}/children", token, parent_page_id, append_payload)
    return page


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a local session Markdown file to Notion.")
    parser.add_argument("markdown_path", help="Path to the session Markdown file.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate locally without calling Notion API.")
    return parser.parse_args()


def main() -> int:
    configure_output_encoding()
    args = parse_args()
    root = project_root()
    markdown_path = Path(args.markdown_path)
    if not markdown_path.is_absolute():
        markdown_path = root / markdown_path
    if not markdown_path.exists():
        print(f"Markdown file not found: {markdown_path}", file=sys.stderr)
        return 2

    markdown = markdown_path.read_text(encoding="utf-8")
    title, body = parse_markdown_title(markdown)
    blocks = markdown_to_notion_blocks(body)
    digest = sha256_file(markdown_path)
    log = load_publish_log(root)
    existing = find_existing_entry(log, digest)

    if args.dry_run:
        print("Notion publish dry-run")
        print(f"Markdown: {relative_display(markdown_path, root)}")
        print(f"Title: {title}")
        print(f"SHA-256: {digest}")
        print(f"Block count: {len(blocks)}")
        print(f"Block batches: {len(batches(blocks, BLOCK_BATCH_SIZE))}")
        if existing:
            print("Duplicate status: already present in publish log")
            if existing.get("page_url"):
                print(f"Existing Notion URL: {existing['page_url']}")
        else:
            print("Duplicate status: not found in publish log")
        return 0

    if existing:
        print("Duplicate upload prevented by SHA-256 publish log.")
        if existing.get("page_url"):
            print(f"Existing Notion URL: {existing['page_url']}")
        return 0

    try:
        env_values = load_root_env(root)
        token = require_env(env_values, "NOTION_TOKEN")
        parent_page_id = require_env(env_values, "NOTION_PARENT_PAGE_ID")
        page = create_page(token, parent_page_id, title, blocks)
    except PublishError as exc:
        print(f"Notion publish failed: {exc}", file=sys.stderr)
        return 1

    page_url = page.get("url")
    if not page_url:
        print("Notion publish failed: response did not include page URL", file=sys.stderr)
        return 1

    entry = {
        "sha256": digest,
        "markdown_path": relative_display(markdown_path, root),
        "title": title,
        "page_url": page_url,
        "published_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notion_version": NOTION_VERSION,
        "block_count": len(blocks),
    }
    log.setdefault("entries", []).append(entry)
    save_publish_log(root, log)

    print(f"Published local session log: {markdown_path}")
    print(f"Notion page URL: {page_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


