# Structural Bass Blind Annotation v0

This local page presents exactly the existing 100 candidates as two blind, deterministic seed-42 sets. It uses MIDI note-on pitch/onset structure only. Set A should be annotated first; Set B is reserved for later holdout review.

## Run

From the server host, start the existing development container server (CPU-only):

```bash
docker exec -it ilkyun-marg-pedaling-dev python /workspace/project/scripts/serve_structural_bass_annotation.py --host 0.0.0.0 --port 8765
```

Forward port `8765` in VS Code's **Ports** view, then open:

```text
http://localhost:8765
```

No package installation, GPU/CUDA, model inference, audio, or MIDI-file access is used.

## Annotate Set A

Choose **Annotate Set A** on the start screen. One candidate is shown at a time. The piano roll contains four preceding groups, the candidate, and eight following groups. Candidate attacks are orange; its lowest pitch is ringed and filled in red.

Keyboard shortcuts:

- `1`: Structural Bass
- `2`: Not Structural Bass
- `3`: Ambiguous
- `←`: previous candidate
- `→`: next candidate

Buttons and keyboard labels save immediately and advance. Comments autosave; previous/next navigation allows review and label correction. Set A and B progress are independent.

## Persistence and export

The server atomically updates this persistent file after every label/comment change:

```text
/workspace/project/analysis/structural_bass_annotation_v0/annotation_results.csv
```

Browser refresh or closure does not clear saved work. **Export annotations** downloads the current set as CSV with `review_id,set,annotation,comment`.

## Blind/hidden separation

- Browser-served files live only under `public/`.
- `annotation_master_hidden.csv` preserves the metrics and original candidate mapping but is outside `public/` and is not exposed by the server.
- `annotation_split_manifest.csv` records the deterministic split and balance audit and is likewise not exposed.
- Never open hidden master files during blind annotation.

## Verified split

- Set A: 50 candidates, exactly 10 per piece and 2 per piece/R-stratum
- Set B: 50 candidates, exactly 10 per piece and 2 per piece/R-stratum
- A/B overlap: 0
- Initial labels/comments: blank

Run the smoke test inside the container:

```bash
python -m unittest tests.test_structural_bass_annotation_v0 -v
```
