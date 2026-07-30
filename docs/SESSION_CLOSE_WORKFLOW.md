# Session Close Workflow

This workflow records the end of a research session in a local Markdown file first, then optionally publishes the same content to Notion.

Trigger phrase:

`오케이 오늘 연구 진행상황 총정리해서 노션에 기록해줘`

When this phrase is used, Codex should follow this document and must not create a Git commit.

## Purpose

- Capture what was actually done in the MARG_research project.
- Keep the local VS Code repository as the source of truth.
- Publish a matching Notion child page only after the local Markdown summary exists.
- Avoid leaking Notion tokens, parent page IDs, or other secrets.

## Required Inputs

The summary builder reads only bounded project evidence:

- Current Git status and changed files.
- `docs/EXPERIMENT_LOG.md`.
- `docs/DECISIONS.md`.
- `docs/NEXT_SESSION.md`.
- `docs/TROUBLESHOOTING.md` for open/watch items.
- Recent small `metadata*.json` files under bounded project paths.

It must not scan large third-party, checkpoint, or generated media files.

## Local Summary Format

Session logs are saved as:

`docs/session_logs/YYYY-MM-DD_<slug>.md`

Each log uses this structure:

```markdown
# YYYY-MM-DD — 세션 제목

## 오늘의 목표
## 완료한 작업
## 검증된 결과
## 주요 결정
## 발생한 문제와 해결
## 미해결 문제
## 다음 세션 우선순위
## 다음 ChatGPT 대화용 Follow-up Prompt
```

If a fact cannot be confirmed from local project evidence, write `미확인` instead of guessing.

## Environment Variables

Actual Notion credentials are read only from the project-root `.env` file.

Required keys:

```text
NOTION_TOKEN=...
NOTION_PARENT_PAGE_ID=...
```

Use `.env.example` as a template. Never commit real `.env` values.

## Commands

Generate a summary preview without writing a file:

```powershell
python scripts/build_session_summary.py --title "세션 제목" --dry-run
```

Create the local Markdown summary:

```powershell
python scripts/build_session_summary.py --title "세션 제목"
```

Preview Notion conversion without calling the Notion API:

```powershell
python scripts/publish_session_to_notion.py docs/session_logs/YYYY-MM-DD_<slug>.md --dry-run
```

Publish to Notion:

```powershell
python scripts/publish_session_to_notion.py docs/session_logs/YYYY-MM-DD_<slug>.md
```

## Notion Publishing Behavior

- Uses the official Notion API.
- Creates a child page under `NOTION_PARENT_PAGE_ID`.
- Uses the first Markdown H1 as the Notion page title.
- Converts remaining Markdown into heading, paragraph, bulleted list, numbered list, and code block objects.
- Splits long rich text into chunks below Notion limits.
- Sends blocks in batches of at most 100.
- Uses HTTP timeout and limited retry for transient API failures.
- Prints API errors without tokens or full parent page IDs.
- Prints the created Notion page URL on success.

## Duplicate Prevention

`publish_session_to_notion.py` computes SHA-256 for the Markdown file.

If the same SHA-256 already appears in:

`docs/session_logs/.notion_publish_log.json`

then the script stops without creating another Notion page.

The publish log must not contain tokens or `.env` values.

## Security Rules

- Do not print `NOTION_TOKEN`.
- Do not print the full `NOTION_PARENT_PAGE_ID`.
- Do not store secrets in docs, source code, examples, fixtures, or publish logs.
- Keep `.env` ignored by Git.
- Use placeholders only in `.env.example`.

## Manual Checklist Before Real Upload

1. Confirm `.env` exists in the project root and contains the two required keys.
2. Confirm the Notion integration has access to the target parent page.
3. Run summary dry-run and inspect the Markdown.
4. Create the local Markdown summary.
5. Run Notion upload dry-run and inspect block count and duplicate status.
6. Run the real upload command only after the local summary looks correct.
7. Do not commit from Codex.
