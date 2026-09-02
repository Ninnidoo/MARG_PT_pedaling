"use strict";

const ALLOWED_LABELS = ["STRUCTURAL_BASS", "NOT_STRUCTURAL_BASS", "AMBIGUOUS"];
const state = {
  activeSet: null,
  candidates: [],
  annotations: new Map(),
  index: 0,
  commentTimer: null,
  saveSequence: 0,
  saveChain: Promise.resolve(),
};

const el = {};

document.addEventListener("DOMContentLoaded", async () => {
  Object.assign(el, {
    landing: document.querySelector("#landing"),
    review: document.querySelector("#review"),
    progress: document.querySelector("#progress"),
    position: document.querySelector("#candidate-position"),
    piece: document.querySelector("#piece-title"),
    pitch: document.querySelector("#candidate-pitch"),
    plot: document.querySelector("#piano-roll"),
    comment: document.querySelector("#comment"),
    saveStatus: document.querySelector("#save-status"),
    previous: document.querySelector("#previous"),
    next: document.querySelector("#next"),
    export: document.querySelector("#export"),
  });

  document.querySelectorAll("[data-set]").forEach((button) => {
    button.addEventListener("click", () => openSet(button.dataset.set));
  });
  document.querySelectorAll("[data-label]").forEach((button) => {
    button.addEventListener("click", () => labelCurrent(button.dataset.label));
  });
  el.previous.addEventListener("click", () => navigate(-1));
  el.next.addEventListener("click", () => navigate(1));
  document.querySelector("#back-to-sets").addEventListener("click", showLanding);
  el.comment.addEventListener("input", scheduleCommentSave);
  el.comment.addEventListener("blur", flushComment);
  document.addEventListener("keydown", handleKeyboard);
  await refreshLandingProgress();
});

async function api(url, options = {}) {
  const response = await fetch(url, {cache: "no-store", ...options});
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

async function refreshLandingProgress() {
  for (const setName of ["A", "B"]) {
    const payload = await api(`/api/results?set=${setName}`);
    const completed = payload.results.filter((row) => row.annotation).length;
    document.querySelector(`#landing-progress-${setName}`).textContent = `${completed} / 50 completed`;
  }
}

async function openSet(setName) {
  if (!/[AB]/.test(setName)) return;
  try {
    const [candidatePayload, resultPayload] = await Promise.all([
      api(`/api/candidates?set=${setName}`),
      api(`/api/results?set=${setName}`),
    ]);
    state.activeSet = setName;
    state.candidates = candidatePayload.candidates;
    state.annotations = new Map(resultPayload.results.map((row) => [row.review_id, row]));
    const firstBlank = state.candidates.findIndex((candidate) => !state.annotations.get(candidate.review_id)?.annotation);
    state.index = firstBlank >= 0 ? firstBlank : 0;
    el.landing.hidden = true;
    el.review.hidden = false;
    el.export.href = `/api/export?set=${setName}`;
    renderCurrent();
  } catch (error) {
    window.alert(`Set ${setName}을 불러오지 못했습니다: ${error.message}`);
  }
}

async function showLanding() {
  await flushComment();
  el.review.hidden = true;
  el.landing.hidden = false;
  state.activeSet = null;
  el.progress.textContent = "";
  await refreshLandingProgress();
}

function currentCandidate() {
  return state.candidates[state.index];
}

function currentResult() {
  const candidate = currentCandidate();
  return state.annotations.get(candidate.review_id) || {
    review_id: candidate.review_id,
    set: candidate.set,
    annotation: "",
    comment: "",
  };
}

function renderCurrent() {
  const candidate = currentCandidate();
  const result = currentResult();
  const completed = [...state.annotations.values()].filter((row) => row.annotation).length;
  el.position.textContent = `Candidate ${candidate.review_id} / 50`;
  el.piece.textContent = `${candidate.composer} — ${candidate.piece_title}`;
  el.pitch.textContent = `Candidate lowest pitch: ${candidate.lowest_note_name} · MIDI ${candidate.lowest_pitch}`;
  el.progress.textContent = `Set ${state.activeSet} · ${completed} / 50 completed`;
  el.comment.value = result.comment || "";
  el.previous.disabled = state.index === 0;
  el.next.disabled = state.index === state.candidates.length - 1;
  document.querySelectorAll("[data-label]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.label === result.annotation);
    button.setAttribute("aria-pressed", button.dataset.label === result.annotation ? "true" : "false");
  });
  renderPianoRoll(candidate);
}

function svgNode(name, attributes = {}, text = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text) node.textContent = text;
  return node;
}

function pitchName(pitch) {
  const names = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

function renderPianoRoll(candidate) {
  const width = 1120;
  const height = 530;
  const margin = {left: 88, right: 32, top: 34, bottom: 62};
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const allPitches = candidate.context.flatMap((group) => group.pitches);
  const pitchMin = Math.min(...allPitches) - 2;
  const pitchMax = Math.max(...allPitches) + 2;
  const maxAbsTime = Math.max(0.25, ...candidate.context.map((group) => Math.abs(group.relative_time_sec))) * 1.08;
  const x = (time) => margin.left + ((time + maxAbsTime) / (2 * maxAbsTime)) * innerWidth;
  const y = (pitch) => margin.top + ((pitchMax - pitch) / (pitchMax - pitchMin)) * innerHeight;
  el.plot.replaceChildren();
  el.plot.append(
    svgNode("title", {id: "plot-title"}, `MIDI pitch and grouped-onset context for ${candidate.review_id}`),
    svgNode("desc", {id: "plot-description"}, "Four preceding grouped onsets, the highlighted candidate, and eight following grouped onsets."),
  );

  const pitchStep = pitchMax - pitchMin > 38 ? 6 : 3;
  for (let pitch = Math.ceil(pitchMin / pitchStep) * pitchStep; pitch <= pitchMax; pitch += pitchStep) {
    el.plot.append(svgNode("line", {x1: margin.left, x2: width - margin.right, y1: y(pitch), y2: y(pitch), class: "grid-line"}));
    el.plot.append(svgNode("text", {x: margin.left - 10, y: y(pitch) + 4, "text-anchor": "end", class: "tick-label"}, `${pitchName(pitch)} · ${pitch}`));
  }
  for (let tick = -2; tick <= 2; tick += 1) {
    const time = tick * maxAbsTime / 2;
    el.plot.append(svgNode("line", {x1: x(time), x2: x(time), y1: margin.top, y2: height - margin.bottom, class: "grid-line"}));
    el.plot.append(svgNode("text", {x: x(time), y: height - margin.bottom + 24, "text-anchor": "middle", class: "tick-label"}, `${time >= 0 ? "+" : ""}${time.toFixed(2)} s`));
  }

  candidate.context.forEach((group) => {
    const candidateGroup = group.relative_group === 0;
    el.plot.append(svgNode("line", {
      x1: x(group.relative_time_sec), x2: x(group.relative_time_sec),
      y1: margin.top, y2: height - margin.bottom,
      class: candidateGroup ? "candidate-line" : "onset-line",
    }));
    el.plot.append(svgNode("text", {
      x: x(group.relative_time_sec), y: margin.top - 10, "text-anchor": "middle", class: "group-label",
    }, group.relative_group > 0 ? `+${group.relative_group}` : String(group.relative_group)));
    group.pitches.forEach((pitch) => {
      const lowest = candidateGroup && pitch === candidate.lowest_pitch;
      if (lowest) {
        el.plot.append(svgNode("circle", {cx: x(group.relative_time_sec), cy: y(pitch), r: 12, class: "candidate-low-halo"}));
      }
      const note = svgNode("circle", {
        cx: x(group.relative_time_sec), cy: y(pitch),
        r: lowest ? 7 : candidateGroup ? 6 : 5,
        class: lowest ? "candidate-low" : candidateGroup ? "candidate-note" : "note",
      });
      note.append(svgNode("title", {}, `${pitchName(pitch)} · MIDI ${pitch} · ${group.relative_time_sec >= 0 ? "+" : ""}${group.relative_time_sec.toFixed(3)} s`));
      el.plot.append(note);
    });
  });
  el.plot.append(svgNode("text", {x: margin.left + innerWidth / 2, y: height - 12, "text-anchor": "middle", class: "axis-label"}, "Relative time from candidate onset (seconds)"));
  el.plot.append(svgNode("text", {x: 18, y: margin.top + innerHeight / 2, transform: `rotate(-90 18 ${margin.top + innerHeight / 2})`, "text-anchor": "middle", class: "axis-label"}, "Newly attacked MIDI pitch"));
}

async function saveResult(annotation, comment) {
  const candidate = currentCandidate();
  const sequence = ++state.saveSequence;
  el.saveStatus.textContent = "Saving…";
  const request = async () => {
    const saved = await api("/api/annotation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({review_id: candidate.review_id, annotation, comment}),
    });
    state.annotations.set(saved.review_id, saved);
    if (sequence === state.saveSequence) el.saveStatus.textContent = "Saved";
    return saved;
  };
  state.saveChain = state.saveChain.catch(() => undefined).then(request);
  return state.saveChain;
}

async function labelCurrent(label) {
  if (!ALLOWED_LABELS.includes(label)) return;
  window.clearTimeout(state.commentTimer);
  state.commentTimer = null;
  try {
    await saveResult(label, el.comment.value);
    if (state.index < state.candidates.length - 1) state.index += 1;
    renderCurrent();
  } catch (error) {
    el.saveStatus.textContent = `Save failed: ${error.message}`;
  }
}

function scheduleCommentSave() {
  window.clearTimeout(state.commentTimer);
  el.saveStatus.textContent = "Unsaved comment…";
  state.commentTimer = window.setTimeout(flushComment, 450);
}

async function flushComment() {
  if (!state.activeSet || !currentCandidate()) return;
  window.clearTimeout(state.commentTimer);
  const result = currentResult();
  if ((result.comment || "") === el.comment.value) return;
  try {
    await saveResult(result.annotation || "", el.comment.value);
  } catch (error) {
    el.saveStatus.textContent = `Save failed: ${error.message}`;
  }
}

async function navigate(delta) {
  await flushComment();
  const nextIndex = Math.max(0, Math.min(state.candidates.length - 1, state.index + delta));
  if (nextIndex !== state.index) {
    state.index = nextIndex;
    renderCurrent();
  }
}

function handleKeyboard(event) {
  if (!state.activeSet || event.ctrlKey || event.metaKey || event.altKey) return;
  if (["TEXTAREA", "INPUT"].includes(document.activeElement?.tagName)) return;
  const actions = {
    "1": () => labelCurrent("STRUCTURAL_BASS"),
    "2": () => labelCurrent("NOT_STRUCTURAL_BASS"),
    "3": () => labelCurrent("AMBIGUOUS"),
    ArrowLeft: () => navigate(-1),
    ArrowRight: () => navigate(1),
  };
  if (actions[event.key]) {
    event.preventDefault();
    actions[event.key]();
  }
}
