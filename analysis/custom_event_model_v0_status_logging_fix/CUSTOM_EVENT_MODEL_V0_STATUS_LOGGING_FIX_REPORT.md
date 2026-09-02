# Custom Event Model v0 Status Logging Fix

## Scope

This is an infrastructure-only correction. Tokenizer/cache, note alignment, PT sequence, architecture, loss, event weights, optimizer, learning rates, batch/accumulation, gradient clipping, AMP recovery policy, ownership, and seed remain unchanged. ASAP test access and Repedal execution are both zero. The failed aligned-v1 run was inspected read-only and was not resumed.

## Exact failure reconstruction

The last committed status at optimizer attempt/step 2000 is valid strict JSON. Its relevant values are:

| Field | Value |
|---|---:|
| `latest_train_total` | 0.9036892784626746 |
| `best_val_total` | null |
| `latest_val_total` | null |
| `latest_validation_components` | null |
| AMP overflow count | 0 |

The traceback can only arise at the status write inside the AMP-overflow branch. On the following optimizer attempt (inferred attempt 2001), the runner had passed its finite forward/objective assertion, unscaled gradients, obtained a non-finite total gradient norm, ran `scaler.step()` and `scaler.update()`, and constructed:

```text
last_overflow.gradient_norm = +inf
```

That field was the exact JSON offender. The local `best_val` and `early_best` variables were initialized to `+inf`, but neither was copied into the failing status; the on-disk status already represented pre-validation values as null. The overflow scale transition was not committed because status writing occurred before the `AMP_OVERFLOW` log line, so it is intentionally not claimed as artifact-confirmed.

## Numerical-health conclusion

- 2,000 successful optimizer steps were recorded.
- Every micro-batch reaching the overflow branch had passed explicit finite checks for all model outputs and objective components.
- The last committed total was finite at 0.903689278.
- The failure event was a non-finite unscaled gradient norm, which is exactly the configured recoverable AMP-overflow case.
- `scaler.step/update` executed before the logging exception; the successful-step counter was not advanced for the overflow.
- There is no evidence of loss/model divergence. However, because no checkpoint exists and the post-overflow parameter-corruption check was located after the failed status write, post-event tensors cannot be directly re-audited from artifacts.

The run therefore failed because an expected AMP diagnostic was not JSON-representable, not because a forward loss became non-finite.

## Logging changes

`src/stage2_event_model/status_logging.py` now owns run-status persistence:

- strict JSON remains `allow_nan=False`;
- pre-validation `best_epoch`, `best_val_total`, `latest_val_total`, and validation components are null;
- internal best/early-best state now starts as `None`, with comparison logic preserving the previous first-validation behavior;
- expected AMP overflow norms serialize as `gradient_norm: null`, `gradient_norm_is_finite: false`, and an explicit `gradient_norm_nonfinite_kind`;
- actual loss/validation/status NaN or Infinity raises `NonFiniteStatusMetricError` during normal operation;
- failure-mode status explicitly lists any non-finite paths rather than hiding them;
- JSON is fully encoded before a temporary file is opened, then written, flushed, fsynced, atomically replaced, and directory-fsynced;
- failure status retains the original exception and traceback; if even that write fails, `status_emergency.log` preserves the primary traceback and the caller re-raises the original exception.

The overflow branch also now updates attempt/step/latest finite loss before persisting the event. Scientific AMP step/skip behavior is unchanged.

## Checkpoint audit

The failed run contains no `best.pt`, `last.pt`, or periodic checkpoint. Therefore no model, optimizer, GradScaler, or RNG state is available and there is no usable resume point. Resume remains forbidden.

Future checkpoint writes occur only after validation. Metadata now requires a non-null best epoch, finite best validation total, and finite GradScaler scale. Model/optimizer/scaler tensors are never sanitized or modified by logging code.

## Verification

- Status-specific serialization tests: 9/9 PASS.
- Combined status, alignment, event-model, ownership/loss, and full-training-helper regression tests: 31/31 PASS.
- Attempt-2000-like pre-validation status: strict JSON PASS.
- Legacy benign best-value `+inf` compatibility: explicit null PASS.
- Unexpected loss `+inf` and NaN: fail-fast PASS.
- Existing-file atomicity under serialization failure: PASS.
- Failure handler primary-traceback preservation and emergency fallback: PASS.
- One real aligned training batch: forward/loss/backward/gradients finite; total 3.374764681; pre-validation status strict JSON read-back PASS; optimizer created false, optimizer/training steps zero.

## Questions

**Q1. Which field was infinite?** `last_overflow.gradient_norm` was positive infinity.

**Q2. Sentinel or numerical failure?** It was not the best-validation sentinel. It was the expected diagnostic value of a recoverable AMP gradient overflow. The old internal best sentinel was benign and unrelated.

**Q3. Healthy through step 2000?** Yes: 2,000 applied optimizer steps and finite logged losses. The next forward/objective was also finite; only its unscaled gradient norm overflowed.

**Q4. Why at attempt 2000?** Attempt 2000 was the last successful logged update. The immediately following overflow attempted to serialize `+inf` before writing its AMP log.

**Q5. Why did the handler fail again?** It reused the same status dictionary containing the non-finite overflow norm and called the same strict serializer.

**Q6. Pre-validation strict JSON now valid?** Yes; undefined validation fields are null.

**Q7. Unexpected NaN/Inf still fail-fast?** Yes. Only the explicitly modeled overflow diagnostic and legacy undefined-best compatibility receive structured encoding.

**Q8. Primary exception preserved?** Yes, including when failure-status persistence itself fails.

**Q9. Usable checkpoint?** No checkpoint exists; resume is impossible and forbidden.

**Q10. Scientific behavior changed?** No. Only status/checkpoint metadata representation and failure logging changed; `None` comparisons are semantically identical to the prior pre-validation `+inf` initialization.

**Q11. Readiness:** **READY FOR FRESH FULL TRAINING RESTART**.

No full training was restarted during this task.
