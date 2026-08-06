# 연구실 공용 GPU 서버 작업 흐름

이 문서는 Pianist Transformer 기반 sustain-pedal 및 repedaling refinement
연구를 공용 GPU 서버에서 안전하게 수행하기 위한 사람용 운영 안내서다.
Docker의 구체적인 빌드·실행 참고 명령은
`docs/SERVER_DOCKER_SETUP.md`를 함께 확인한다.

## 핵심 원칙

- 호스트는 Git, 소스 파일, 외부 저장소, Docker 수명 주기를 관리한다.
- Python 의존성이 필요한 실행은 Docker 컨테이너 안에서만 한다.
- public 데이터는 읽기 전용으로, 개인 결과는 private 경로에 저장한다.
- 컨테이너에는 할당받은 GPU 한 장만 노출한다.
- 공용 서버의 다른 사용자 Docker 자원은 조회 외에 건드리지 않는다.
- 분석 → 계획 → 최소 수정 → 검증 → 보고 순서를 지킨다.
- 큰 학습, 장시간 GPU 작업, 환경 교체, 위험한 명령은 사전 승인을 받는다.

## 현재 상태

2026-07-30에 읽기 전용으로 확인한 현재 실행 환경은 다음과 같다.

| 항목 | 현재 확인값 |
| --- | --- |
| 컨테이너 이름 | `ilkyun-marg-pedaling-dev` |
| 상태 | 실행 중 |
| 이미지 | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime` |
| 작업 디렉터리 | `/workspace/project` |
| 주 프로세스 | `sleep infinity` |
| 노출 GPU | RTX 2080 Ti 한 장 |
| 컨테이너 내부 GPU | index 0, UUID `GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b` |

이 값은 현재 런타임에 대한 기록이며 영구 보장이 아니다. 작업 시작 전
`docker ps`, `docker inspect`, `nvidia-smi` 같은 읽기 전용 명령으로 다시
확인한다.

현재 컨테이너와 저장소의 재현용 설정에는 차이가 있다.
`.devcontainer/devcontainer.json`은 PyTorch 2.7.1/CUDA 11.8 이미지와 다른
GPU UUID를 지정하지만, 현재 컨테이너는 위 표의 PyTorch 2.8.0/CUDA 12.6
이미지와 GPU로 실행 중이다. 이 문서 작업에서는 어느 쪽도 변경하지
않았다. 빌드·재생성 전에 원인을 확인하고 사용자 승인을 받아 일치시킨다.

## 호스트와 컨테이너의 역할

| 작업 | 실행 위치 | 비고 |
| --- | --- | --- |
| Git 상태 확인, diff, add, commit, pull, push | 호스트 | 사용자가 Remote SSH 호스트에서 수행 |
| 프로젝트 파일 편집 | 호스트 또는 Dev Container | 실제 파일은 project bind mount에 유지 |
| Docker 상태 조회와 수명 주기 관리 | 호스트 | 생성·중지·삭제·재생성은 승인 필요 |
| Python, pip, PyTorch | 컨테이너 | 호스트 설치 금지 |
| 데이터 전처리, 분석, 테스트 | 컨테이너 | 프로젝트 의존성이 필요할 때 |
| 추론, 평가, 승인된 제한 학습 | 컨테이너 | 단일 GPU와 persistent output 사용 |
| 외부 저장소 clone/update | 호스트 | `third_party` 관리 규칙 준수 |

호스트에서 `pip`, `conda`, `apt`, `sudo`로 패키지를 설치하지 않는다.
컨테이너 안의 설치도 의존성 변경이므로 먼저 승인을 받고 버전을 기록한다.

## 경로와 bind mount

| 용도 | 호스트 경로 | 컨테이너 경로 | 모드 |
| --- | --- | --- | --- |
| 프로젝트 | `/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling` | `/workspace/project` | read/write |
| 공개 데이터·공개 pretrained model | `/public/intern_2026_summer_public_dataset/ilkyun_data` | `/workspace/public` | read-only |
| 개인 checkpoint·output·log·cache | `/private/intern_2026_summer_private_dataset/ilkyun_data` | `/workspace/private` | read/write |

bind mount의 실제 데이터는 호스트에 남으므로 컨테이너가 중지되거나
삭제되어도 유지된다. 실험 결과는 가능하면 다음처럼 private 아래에 둔다.

    /workspace/private/checkpoints
    /workspace/private/outputs
    /workspace/private/logs
    /workspace/private/runs
    /workspace/private/cache

공유 public 경로는 수정하지 않는다. 연구 코드에는 호스트 절대 경로를
하드코딩하지 말고 `/workspace/project`, `/workspace/public`,
`/workspace/private` 또는 대응 환경 변수와 CLI 인자를 사용한다.
Docker/Dev Container 인프라 파일에는 서버 bind mount를 위해 호스트
절대 경로를 명시할 수 있다.

다음 항목은 이미지에 `COPY`하거나 Git에 추가하지 않는다.

- dataset과 원본 MIDI 모음
- pretrained model과 개인 checkpoint
- 생성 output, log, run, cache
- `.env`, 비밀번호, API key, token
- SSH key, Git 설정, Git credential

Docker socket, 호스트 홈 디렉터리, SSH/Git 인증 파일도 컨테이너에
마운트하지 않는다.

## 현재 컨테이너 접속

터미널 접속:

    docker exec -it ilkyun-marg-pedaling-dev bash

VS Code 접속:

1. VS Code에서 Remote SSH로 서버에 접속한다.
2. 호스트의
   `/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling`을 연다.
3. 현재 컨테이너를 그대로 쓸 때는
   `Dev Containers: Attach to Running Container...`를 선택한다.
4. `ilkyun-marg-pedaling-dev`를 선택하고 `/workspace/project`을 연다.
5. Git 작업은 Dev Container가 아니라 Remote SSH 호스트 창이나 호스트
   터미널에서 수행한다.

`Dev Containers: Reopen in Container`는 설정에 따라 빌드 또는 새
컨테이너 생성을 유발할 수 있다. 현재 컨테이너를 유지해야 할 때는
승인 없이 실행하지 않는다.

## 단일 GPU 규칙

현재 컨테이너의 Docker device request는 호스트 GPU 하나만 지정하며,
컨테이너 안에서는 RTX 2080 Ti 한 장이 index 0으로 보이는 것이 확인됐다.

작업 전 호스트에서 확인:

    nvidia-smi

현재 컨테이너 내부 확인:

    docker exec ilkyun-marg-pedaling-dev nvidia-smi -L

PyTorch 작업 전에 컨테이너 안에서 확인할 기대값:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"

기대 조건은 CUDA 사용 가능, device count 1, RTX 2080 Ti다. 물리 GPU를
하나 골랐더라도 컨테이너 내부에서는 `cuda:0`으로 보인다.
`--gpus all`을 사용하지 않는다. RTX 2080 Ti에서는 Colab의 BF16 경로를
가정하지 말고, 검증된 코드 경로에서 FP16을 사용한다. GPU 교체,
multi-GPU, 큰 학습 또는 장시간 점유는 먼저 승인과 서버 할당 확인이
필요하다.

## Git과 third-party 관리

- 메인 저장소의 Git 작업은 호스트에서만 한다.
- Codex는 Git commit/push를 하지 않는다.
- 컨테이너에서 GitHub 로그인을 새로 만들지 않는다.
- 전역·시스템 Git 설정을 변경하지 않는다.
- SSH key, `.gitconfig`, credential store를 컨테이너로 복사하지 않는다.
- 데이터, checkpoint, output, log, cache, secret은 commit하지 않는다.
- commit 전에 `git status --short`와 `git diff --check`로 범위를 확인한다.

`third_party/PianistTransformer`는 외부 저장소다.

- 메인 저장소의 `.gitignore`로 추적하지 않는다.
- clone, fetch, checkout, commit 확인은 호스트에서 별도로 수행한다.
- 검증된 기준 commit은
  `747df2d12291e37f6638b39f1b71517e579ad48c`이다.
- 가능하면 외부 코드를 수정하지 않는다.
- 수정이 필요하면 기준 commit, 이유, patch, 영향받는 실험을 문서화한다.
- 외부 저장소 내용을 Docker 이미지에 복제하거나 메인 저장소에 vendor하지
  않는다.

## Codex 표준 작업 순서

1. `AGENTS.md`와 관련 docs, 코드, 현재 상태를 읽는다.
2. `git status` 및 필요한 런타임 상태를 읽기 전용으로 확인한다.
3. 목표, 영향 파일, 위험, 검증 방법을 정리한다.
4. 승인 범위 안에서 가장 작고 되돌리기 쉬운 수정만 한다.
5. 관련 테스트 또는 정적 검사를 실행한다.
6. 예상 artifact가 persistent 경로에 생성됐는지 확인한다.
7. 변경 파일, 실제 실행 명령, 결과, 경고, 미확정 사항을 보고한다.

분석·진단·검토 요청은 자동으로 수정 권한을 뜻하지 않는다. 기존 사용자
변경과 관련 없는 dirty worktree 항목을 보존한다. 테스트가 불가능하면
성공했다고 쓰지 말고 미검증 이유를 명시한다. 추론 완료는 비어 있지 않은
MIDI output과 그 persistent 경로를 확인했을 때만 선언한다.

## 명령 정책

### 읽기 전용 또는 통상적인 확인

대상을 정확히 제한한 다음 다음과 같은 확인을 수행할 수 있다.

- `pwd`, `ls`, `find`, `rg`, 파일 읽기
- `git status`, `git diff`, `git log`
- `docker ps`, `docker inspect`, `docker logs`
- `nvidia-smi`
- 현재 컨테이너에서의 작은 정적 검사와 승인 범위의 테스트

### 사전 승인이 필요한 작업

- `docker build`, 새 `docker run`, container recreate
- 컨테이너 start/stop/restart/remove/rename
- image 또는 volume 생성·삭제·교체
- Python·PyTorch·CUDA·base image·checkpoint 버전 변경
- 컨테이너 내부 package 설치 또는 upgrade
- 큰 dataset/model 다운로드
- 장시간 GPU 점유, 전체 학습, multi-GPU 실행
- 대규모 rewrite, 데이터 변환, 덮어쓰기 또는 삭제

### 금지 작업

- `docker system prune`
- `docker image prune`
- `docker container prune`
- `docker volume prune`
- 다른 사용자의 container, image, volume, network 수정 또는 삭제
- 호스트 package 설치와 `sudo`
- 범위가 넓거나 대상이 불명확한 `rm -rf`
- `--gpus all`
- Docker socket, host home, SSH/Git credential mount
- privileged mode, host network, host PID, host IPC
- secret, dataset, checkpoint, output을 이미지나 Git에 포함

Docker 자원을 변경해야 한다면 먼저 이름과 owner/project label을
읽기 전용으로 확인한다. label이 없는 현재 컨테이너는 이름과 mount 등
여러 근거로 소유 범위를 확인하고, 변경은 반드시 사용자 승인 후 수행한다.

## 연구 방향

### 기준선

공식 checkpoint `yhj137/pianist-transformer-rendering`와 Pianist
Transformer commit `747df2d12291e37f6638b39f1b71517e579ad48c`을 기준으로
한다. 성공한 Colab GPU smoke test는 검증된 참조 workflow로 보존한다.
새로운 서버 실험은 Docker 환경에서 수행하되 참조 notebook을 깨뜨리지
않는다.

모든 실험에는 seed, input, model commit, checkpoint, 생성 parameter,
runtime, output path와 pedal metric을 기록한다.

### Stage 1: pedal-tokenizer information-loss analysis

human raw CC64를 source of truth로 두고 official tokenizer가 pedal depth,
transition timing, repedaling 정보를 얼마나 보존·손실하는지 측정한다.

- 입력 166개 중 165개 분석, CC64 없는 `3-1.mid` 제외
- raw repedal candidate 15,549개
- preserved 13,256개, lost 2,293개
- micro recall 0.8525, macro recall 0.8288

이 수치는 tokenizer의 정보 손실을 보여주는 탐색 결과다. candidate를
최종 repedaling label로 해석하지 않는다.

### Stage 2: event-based pedal-target audit

unique-note-onset IOI마다 최대 두 개의 UP/DOWN transition slot, 정규화된
event time `tau`, 별도의 pedal-ON depth target을 검토한다.

- 165개 파일에서 two-slot lossless coverage 0.998761
- overflow IOI 697개
- tokenizer, model prediction, training, GPU/CUDA를 사용하지 않은 target
  audit
- depth 경계, hysteresis, rapid reversal 처리, annotation protocol은 미확정

다음 단계는 작은 target prototype 검증, 연구 질문 구체화, annotation
protocol 확정, pedal-heavy 예시 선별이다. 이 결정 전에 exploratory
metric을 결론으로 포장하거나 전체 모델 학습을 시작하지 않는다.

근거 문서는 `docs/EXPERIMENT_LOG.md`,
`analysis/pedal_tokenizer_analysis/HUMAN_BATCH_REPORT.md`,
`analysis/stage2_pedal_target_audit/summary.md`를 참고한다.

## 작업 시작 체크리스트

- [ ] `AGENTS.md`와 이 문서를 읽었다.
- [ ] 호스트 `git status`로 기존 변경을 확인했다.
- [ ] 현재 container 이름, image, mount를 읽기 전용으로 확인했다.
- [ ] 할당 GPU와 컨테이너 내부 device count 1을 확인했다.
- [ ] 입력은 public read-only, 결과는 private persistent 경로로 정했다.
- [ ] 실행 규모와 package/version 변경의 승인 필요 여부를 판단했다.
- [ ] third-party commit과 checkpoint를 기록할 준비가 됐다.

## 작업 종료 체크리스트

- [ ] 관련 테스트 또는 정적 검사 결과와 exit status를 기록했다.
- [ ] output/checkpoint/log가 `/workspace/private` 아래에 남는지 확인했다.
- [ ] dataset/checkpoint/output/secret이 Git 대상이 아닌지 확인했다.
- [ ] 변경 파일과 기존 미관련 변경을 구분했다.
- [ ] 실제 실행 명령만 검증 완료로 문서화했다.
- [ ] 경고, 실패, 미확정 연구 결정을 명시했다.
- [ ] Git commit/push와 Docker 수명 주기 변경을 자동 실행하지 않았다.
