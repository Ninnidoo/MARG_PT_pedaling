#!/usr/bin/env bash
# Sequentially invoke the already-validated one-score listening pipeline.
set -uo pipefail

repo=/workspace/project
scores="$repo/third_party/PianistTransformer/data/midis/testset/score"
outputs="$repo/outputs"
logdir="$repo/logs/listening_comparison_5class_all_v0"
resultdir="$logdir/results"
summary="$logdir/batch_summary.json"

mkdir -p "$logdir" "$resultdir" "$outputs/midi" "$outputs/audio"

mapfile -t score_paths < <(find "$scores" -maxdepth 1 -type f \( -iname '*.mid' -o -iname '*.midi' \) -printf '%f\n' | LC_ALL=C sort)
total=0
for name in "${score_paths[@]}"; do
  [[ "$name" == "0.mid" ]] && continue
  total=$((total + 1))
  score="$scores/$name"
  stem="${name%.*}"
  stdout="$logdir/${stem}.stdout.json"
  stderr="$logdir/${stem}.stderr.log"
  record="$resultdir/${stem}.json"
  started="$(date --iso-8601=seconds)"

  echo "[$started] START score=$name" | tee -a "$logdir/runner.log"
  if python -m src.listening_comparison_5class_v0 \
      --score "$score" \
      --output-midi-dir "$outputs/midi" \
      --output-audio-dir "$outputs/audio" \
      --seed 42 >"$stdout" 2>"$stderr"; then
    python - "$stdout" "$record" "$name" <<'PY'
import hashlib, json, sys
from pathlib import Path

stdout, record, score_name = map(Path, sys.argv[1:])
data = json.loads(stdout.read_text(encoding="utf-8"))
outputs = data["outputs"]
paths = {key: Path(value) for key, value in outputs.items()}
record_data = {
    "score": score_name.name,
    "status": "success",
    "outputs": outputs,
    "four_outputs_nonempty": all(path.is_file() and path.stat().st_size > 0 for path in paths.values()),
    "non_pedal_verification_passed": bool(data["non_pedal_midi_verification"]["passed"]),
    "stage2_decoded_representatives": data["stage2"]["inference"]["decoded_representatives"],
    "stage2_all_slots_zero": data["stage2"]["inference"]["decoded_representatives"] == [0],
    "midi_byte_identical": (
        hashlib.sha256(paths["original_pt_midi"].read_bytes()).digest()
        == hashlib.sha256(paths["stage2_5class_midi"].read_bytes()).digest()
    ),
    "run_info": data["run_info"],
}
record.write_text(json.dumps(record_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    echo "[$(date --iso-8601=seconds)] SUCCESS score=$name" | tee -a "$logdir/runner.log"
  else
    python - "$record" "$name" "$started" "$stderr" <<'PY'
import json, sys
from pathlib import Path

record, score_name, started, stderr = map(Path, sys.argv[1:])
message = stderr.read_text(encoding="utf-8", errors="replace")[-12000:]
record.write_text(json.dumps({
    "score": score_name.name,
    "status": "failed",
    "started": str(started),
    "failure_stage": "one_score_pipeline",
    "error_tail": message,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    echo "[$(date --iso-8601=seconds)] FAILURE score=$name; continuing" | tee -a "$logdir/runner.log"
  fi
done

python - "$resultdir" "$summary" "$total" <<'PY'
import json, sys
from pathlib import Path

resultdir, summary, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(resultdir.glob("*.json"))]
successes = [item for item in records if item["status"] == "success"]
failures = [item for item in records if item["status"] == "failed"]
summary.write_text(json.dumps({
    "target_scores": expected,
    "excluded_scores": ["0.mid"],
    "successful_scores": [item["score"] for item in successes],
    "successful_score_count": len(successes),
    "failed_scores": [{key: item.get(key) for key in ("score", "failure_stage", "error_tail")} for item in failures],
    "failed_score_count": len(failures),
    "four_outputs_generated_score_count": sum(item.get("four_outputs_nonempty", False) for item in successes),
    "non_pedal_verification_passed_count": sum(item.get("non_pedal_verification_passed", False) for item in successes),
    "byte_identical_midi_scores": [item["score"] for item in successes if item.get("midi_byte_identical")],
    "stage2_all_slots_zero_scores": [item["score"] for item in successes if item.get("stage2_all_slots_zero")],
    "score_outputs": {item["score"]: item["outputs"] for item in successes},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[$(date --iso-8601=seconds)] COMPLETE target_scores=$total summary=$summary" | tee -a "$logdir/runner.log"
