# Prompt 001: Project Structure

## Purpose

Create the baseline repository structure for the MARG_research Pianist Transformer repedaling refinement project.

## Prompt

```text
먼저 프로젝트 루트의 AGENTS.md를 읽고 모든 지침을 따라줘.

현재 열린 MARG_research 프로젝트의 기본 구조를 만들어줘.

수행할 작업:
1. 다음 폴더를 만든다.
   - docs
   - docs/prompts
   - third_party
   - src/repedaling
   - scripts
   - data/raw
   - data/interim
   - data/processed
   - checkpoints
   - outputs/midi
   - outputs/audio
   - outputs/figures
   - experiments
   - notebooks
   - configs
   - tests
   - logs

2. 루트에 다음 파일을 만든다.
   - README.md
   - .gitignore

3. docs 안에 다음 문서의 기본 틀을 만든다.
   - SETUP.md
   - USAGE.md
   - RESEARCH_PLAN.md
   - EXPERIMENT_LOG.md
   - TROUBLESHOOTING.md
   - DECISIONS.md

4. docs/prompts 안에 다음 파일을 만든다.
   - 001_project_structure.md

5. 빈 폴더가 Git에서 유지되도록 필요한 곳에 .gitkeep 파일을 둔다.

.gitignore에는 최소한 다음을 포함해줘:
- .venv/
- **pycache**/
- *.pyc
- checkpoints/*
- outputs/*
- logs/*
- data/raw/*
- data/interim/*
- .env
- .vscode/
- *.ckpt
- *.pt
- *.pth
- *.bin
- *.wav
- *.flac

단:
- 각 폴더의 .gitkeep은 무시되지 않게 한다.
- data/raw와 checkpoints의 실제 데이터 파일은 Git에 포함하지 않는다.
- AGENTS.md는 수정하지 않는다.

README.md에는 다음만 간단히 작성해줘:
- 프로젝트 이름
- Pianist Transformer 기반 repedaling refinement 연구라는 목적
- 현재 단계가 baseline inference 환경 구축이라는 점
- 주요 폴더 설명
- 아직 모델 설치와 inference가 완료되지 않았다는 점

각 docs 문서에는 제목과 최소한의 섹션 틀만 작성해줘.
아직 확인되지 않은 명령이나 사실을 성공한 것처럼 쓰지 마.

docs/prompts/001_project_structure.md에는:
- 이 작업의 목적
- 내가 지금 보낸 프롬프트 전문
- 결과를 나중에 적을 수 있는 Result 섹션
을 작성해줘.

아직 다음 작업은 하지 마:
- Pianist Transformer 저장소 clone
- Python 가상환경 생성
- 패키지 설치
- checkpoint 다운로드
- 모델 실행
- Git commit
- Git global 또는 system 설정 변경

완료 후:
1. 생성하거나 수정한 파일 목록
2. 최종 폴더 구조
3. git status 결과
4. unresolved issue
를 요약해줘.
```

## Result

To be filled in after the structure is verified.