# Custom Event Tokenizer Specification v0

- Raw source: original performance MIDI CC64 timestamp/value messages; Pedal1–4 samples are never used to reconstruct events.
- Parser ordering: `per-track file order preserved; equal-tick global deterministic key=(tick,track_index,message_index); last effective CC64 state wins`. Cross-track equal-tick ordering is deterministic for audit repeatability but semantically ambiguous.
- States: ZERO 0–25, LOW 26–63, HALF 64–103, FULL 104–127; representatives `[0, 51, 79, 127]`.
- Vocabulary: `NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL` (absolute destination state).
- Equal timestamp: stable source order, then last effective CC64 state; quantized same-state changes are suppressed.
- Timeline: distinct non-pedal note onsets; right-open `[t_i,t_(i+1))`; event at an onset belongs to that onset's interval with tau=0.
- Initial state: state immediately before the first onset (`event_time < first_onset`); MIDI default ZERO.
- Terminal interval: `[last_onset, latest_note_off]`; later events are unsupported and audited.
- Tau: exact raw float ratio, no discretization.
- Slots: 0→NONE/NONE; 1→E1/NONE; 2→E1/E2; odd 3+→last/NONE; even 4+→penultimate/last.
- Decoder: deterministic chronological SET application, no smoothing/snapping/randomization.
- Initial serialization: representative CC at `max(0, first_onset-1)` and ordered before any tau=0 SET/note-on at tick zero.
- Training/model/optimizer/checkpoint activity: zero.
- ASAP test MIDI/cache/candidate access: zero. Repedal executions: zero.
