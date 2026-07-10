# MARG Research

Pianist Transformer 기반 repedaling refinement 연구 프로젝트입니다.

## Current Stage

현재 단계는 baseline inference 환경 구축입니다. 아직 모델 설치와 inference는 완료되지 않았습니다.

## Purpose

Pianist Transformer를 baseline으로 사용해 sustain pedal 및 repedaling refinement를 연구합니다.

## Main Folders

- `docs/`: setup, usage, research notes, troubleshooting, decisions
- `docs/prompts/`: project setup and research prompts
- `third_party/`: external repositories such as Pianist Transformer
- `src/repedaling/`: original repedaling refinement code
- `scripts/`: executable utilities and experiment entry points
- `data/`: raw, interim, and processed data
- `checkpoints/`: pretrained and trained model weights, not tracked by Git
- `outputs/`: generated MIDI, audio, figures, and analysis outputs
- `experiments/`: experiment records and configuration snapshots
- `notebooks/`: exploratory analysis notebooks
- `configs/`: project and experiment configuration files
- `tests/`: automated tests
- `logs/`: execution logs, not tracked by Git