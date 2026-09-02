# Full State-Conditioned Binary 2-Slot v0 Report

| item | result |
|---|---|
| Run status | **STOPPED — tiny-overfit FAIL** |
| Multi-window sanity | PASS (4 windows, 1041 owned MAIN onsets) |
| No-grad encoder parameter | `encoder.embed_tokens.weight` (inactive inherited T5 path) |
| Boundary/Main scalar loss ratio | 0.974211 |
| Boundary/Main encoder grad ratio | 1.227165 |
| Boundary/Main head grad ratio | 1.090235 |
| Selected boundary_loss_weight | 0.25 |
| Tiny overfit | FAIL |
| Tiny final Count accuracy | 0.498715 |
| Tiny final state agreement | 0.497429 |
| Tiny final correction fraction | 0.502571 |
| Tiny total-loss reduction | 0.869030 |
| Persistent dominant prediction | N=1: 93.44% (727/778) |
| Full epochs completed | 0 / 10 |
| Validation epochs completed | 0 |
| Best epoch / F1 | N/A — full training not authorized by gate |
| Best / last checkpoint | Not created |
| tmux full-training session | Not launched |
| ASAP test access | 0 |

## Guarded run outcome

Stage A passed CUDA/provenance/checkpoint/compile/unit checks. Stage B passed PRE-once, POST-once, strict owner chronology, zero missing/duplicate supervision, monotonic owner order, and state continuity across all real window boundaries. Stage C completed six independent backward audits with finite gradients. The sole encoder parameter without a gradient was the expected inherited `encoder.embed_tokens.weight`; the custom PT input path supplies `inputs_embeds`, so this path is inactive rather than a missing live gradient.

The automatic Stage D rule selected lambda=0.25 because lambda times max(encoder, head boundary/Main gradient ratio) was 0.306791; lambda=0.5 yielded 0.613582, above the frozen 0.5 ceiling.

The train-only 500-step tiny run was numerically stable and reduced total loss from 4.611375 to 0.603950. Nevertheless, it converged to an N=1-dominant self-reconciliation regime: Count accuracy remained 0.4987, state agreement 0.4974, and correction-required fraction 0.5026. This meets the task's explicit sustained single-class-collapse / failed Count-learning hard-stop condition. Therefore no full-training optimizer step, validation inference, checkpoint, or tmux process was started.

## Engineering fixes made during the guarded stages

- Batched owner windows now sum their loss before one backward, avoiding a second backward through a shared encoder graph. The Binary 2-Slot unit regression was rerun and passed.
- AMP encoder outputs are cast to FP32 for the three FP32 linear heads, preventing half/float dtype mismatch without changing the frozen architecture.
- Read-only cached NumPy boundary inputs are copied before conversion to Torch tensors; scientific values are unchanged.
- The first Stage C failure artifact is retained as a resolved engineering incident; the subsequent Stage B–D run passed.

## Epoch results

| epoch | train loss | count acc | state agreement | correction frac | val P | val R | val F1 | pred/ref transitions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | — | — | — | — | — | — | — | — |

No ASAP test metadata or MIDI was accessed. No test inference or test metric was run. The next required action is a human design review of why hard online reconciliation admits the observed N=1-dominant fixed point; the frozen scientific formulation was not changed automatically.
