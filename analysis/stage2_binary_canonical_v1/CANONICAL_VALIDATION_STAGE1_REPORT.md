# Canonical Validation Stage 1 Report

## 1. Scientific goal

This new, validation-controlled lineage asks whether a binary Stage 2 that changes sustain CC64 only can improve the official Pianist Transformer pedal-distribution metric while Original PT expressive performance is permanently fixed as Stage 1.

## 2. Canonical Stage 1 invariant

All Stage 2 candidates must share these 19 exact Original PT validation performances. Stage 1 neural inference is never rerun after freeze. Stage 2 may transplant CC64 only, and strict non-CC64 equality is a hard pre-metric gate.

## 3. Split and data provenance

- ASAP split: `/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv` SHA-256 `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b`.
- Pieces: train 180, validation 19, test 23.
- Performances: train 892, validation 71, test 104.
- Future training pool: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances.

## 4. Validation score selection (19 pieces)

Existing source-of-truth manifest: `/workspace/project/analysis/stage2_binary_v0/validation_eval_v0/validation_score_manifest.csv` SHA-256 `4aac49a07a5bf0fdac24c9f9c3de7a268455f6a78f3ddb7bcf331ce7817e3711`. Rule: max direct validation-row metadata support; lexical tie-break. No new selection rule was introduced.

| Piece | Composer | Title | Selected score | Support | Score SHA-256 |
|---|---|---|---|---:|---|
| piece_f10784ae79f0f210 | Bach | Prelude_bwv_856 | Bach/Prelude/bwv_856/midi_score.mid | 1/1 | `0de233c5b14f0e66a131e1edb43871fd15f8ce9589315539b7f16ebee0d16a32` |
| piece_08aa5ff2ada461e0 | Bach | Prelude_bwv_862 | Bach/Prelude/bwv_862/midi_score.mid | 1/1 | `58ad7ffa1d46a3a9ed98b97bde159074f4a0b20d81693234a50647d253e28561` |
| piece_b06ab236f236fd76 | Bach | Prelude_bwv_888 | Bach/Prelude/bwv_888/midi_score.mid | 1/1 | `cd5e26851da2f25da1101480f7a49e4a6fa03c6483a90c4fe5695bfd54813543` |
| piece_69862af5096ee3fa | Beethoven | Piano_Sonatas_27-1 | Beethoven/Piano_Sonatas/27-1/midi_score.mid | 7/7 | `b88f24f3421c0786807418dfd195bc496210297ee73e2768a4544ea452bf1d37` |
| piece_ab42b317dcb38fc5 | Beethoven | Piano_Sonatas_29-2 | Beethoven/Piano_Sonatas/29-2/midi_score.mid | 1/1 | `4df10d08fee9fc497924af19fde31f1cfa59ff51fac94613c031cb1e2944fe8b` |
| piece_b45364ab6817194a | Beethoven | Piano_Sonatas_8-1 | Beethoven/Piano_Sonatas/8-1/midi_score.mid | 2/2 | `3ec8ab8df110c2e911bf27127c3eb84bf8e436232f65677b64e005e636f24673` |
| piece_18921bc81d708d16 | Beethoven | Piano_Sonatas_9-3 | Beethoven/Piano_Sonatas/9-3/midi_score.mid | 1/1 | `80929db64cd5f08092c10a9d05b88f23f0bfc57c57076ebc7a67852fdb90b135` |
| piece_db97fbaed2036f5b | Chopin | Etudes_op_25_12 | Chopin/Etudes_op_25/12/midi_score.mid | 9/9 | `54f3ec755fbe2654fd70ece57d24524ec81ab8f68714c9b9cfe5aa851a320657` |
| piece_e7ad71b998a8d477 | Haydn | Keyboard_Sonatas_32-1 | Haydn/Keyboard_Sonatas/32-1_no_repeat/midi_score.mid | 3/5 | `8a2200ccedaf10ba5a52ac14ab13cc4410c57627508f91bf842b23c0b9c297a8` |
| piece_e38184a2e9753bdc | Haydn | Keyboard_Sonatas_39-2 | Haydn/Keyboard_Sonatas/39-2/midi_score.mid | 1/1 | `f3562f963273f2aa7a11b94024310e46fc4e6103e7cb9d461fd9c2f4c93fea68` |
| piece_0a888a7d5fbd06e6 | Haydn | Keyboard_Sonatas_46-1 | Haydn/Keyboard_Sonatas/46-1/midi_score.mid | 5/5 | `27171efb0d9905d7990d863833d989180ab5c448043866b709f325a632e068de` |
| piece_901ec1910fe9fa9e | Liszt | Concert_Etude_S145_2 | Liszt/Concert_Etude_S145/2/midi_score.mid | 4/4 | `bcc471a14fb27b5353c1921cb85fc139a0c6f89af079a33bc1542806bf65121d` |
| piece_4b31472ad4376d9e | Liszt | Gran_Etudes_de_Paganini_6_Theme_and_Variations | Liszt/Gran_Etudes_de_Paganini/6_Theme_and_Variations/midi_score.mid | 3/5 | `5b74ef1d4d6d9ba818a9e849b4bbcd98609eaa08986fb195b8a7a8263ff5eb43` |
| piece_7195bbce81550519 | Liszt | Mephisto_Waltz | Liszt/Mephisto_Waltz/midi_score.mid | 14/14 | `d34b2ef57790b9ab24aaf929db6548c4ff4ead596e71b7cb9de749214c3c1424` |
| piece_de3f82957f1b3532 | Ravel | Pavane | Ravel/Pavane/midi_score.mid | 2/2 | `6dbc6ce9d1b86a005f427a98d3f94d340dc13934aae929e9e26bfb96a30a5e1e` |
| piece_759b324c1cceac54 | Schubert | Piano_Sonatas_664-1 | Schubert/Piano_Sonatas/664-1/midi_score.mid | 4/4 | `8eb27191af65afdd811672a52cb1f5e1612be4910701f4906b5699f57f3437d5` |
| piece_daefdda4e1923cc6 | Schumann | Arabeske | Schumann/Arabeske/midi_score.mid | 2/2 | `a5e18cdc262a7fa3428245b9c83e42615cc9c11def0f2ed3d53720eed834fd5d` |
| piece_3056e93b3e51b646 | Schumann | Kreisleriana_6 | Schumann/Kreisleriana/6/midi_score.mid | 3/3 | `d34cfc532697f5e7e0c8e54067496d41c8c950595180789e49d50f4cf7f6ac9d` |
| piece_555ff6cd3a13efb9 | Scriabin | Etudes_op_8_11 | Scriabin/Etudes_op_8/11/midi_score.mid | 3/3 | `c1cafe265fa4d54e990f4512cf8656c7c03fb3524b5a84dadaa515cc94e8f0aa` |

## 5. Exact Original PT CPU inference provenance

- Checkpoint: `/workspace/project/checkpoints/pianist_transformer` (`8fb9111efc147adcc61a4885bd42a6e6b625626d9fb8576811916927aa8281d0`).
- PT revision: `747df2d12291e37f6638b39f1b71517e579ad48c`.
- Runtime: Python 3.11.13, torch 2.8.0+cu126, transformers 4.54.0.
- Device/dtype: `cpu` / `torch.bfloat16`.
- Attention: argument `omitted (validated historical Transformers default/auto path)`; resolved model/encoder/decoder `sdpa` / `sdpa` / `sdpa`.
- Sampling: seed 42, reset per piece; do_sample=True; temperature=1.0; top-p=0.95; top-k=50; num_beams=1; repetition_penalty=1.0.
- Context/overlap: 4096 tokens / 0.5. Official pitch constraint, overlapping generation, and map_midi are unchanged.
- Inference configuration identifier: `106950cd83fce909a2229a8911ff99d15b6a9f7e1669baf50ccb8a2b8004ac5a`.

## 6. Canonical generation integrity

- **19/19 PASS**. Every piece has generated IDs, dumped/reloaded `original_pt.mid`, and inference metadata.
- Score/output token length and pitch hard constraint were checked; every saved MIDI reloaded; no completed piece is regenerated.

## 7. Canonical hashes

| Piece | Generated IDs SHA-256 | Canonical MIDI SHA-256 | Notes | Duration (s) |
|---|---|---|---:|---:|
| piece_f10784ae79f0f210 | `cdad2e12dfdacab37d299da36e779215ebf248e909863e5c06e49d0edc26f0f9` | `02a845df0b2da237edde2d902bb0f8106ac92124594c5c433d8aa487dd8e19da` | 1045 | 67.946095 |
| piece_08aa5ff2ada461e0 | `a43c65980daadbce91083b6b56da2e2064dfbdb2f7491e8e98f58b765ac059d0` | `f524a18802d89c1256f41b27ea7ea05e6a25e7043dde7f19c7ea87c687f8ff0a` | 696 | 114.136472 |
| piece_b06ab236f236fd76 | `91ed02b094c52803528ce34d4d780046b425df9cc7a3f52c5631537246bc797b` | `a0d5cd9fb69e24f836da07d4c9e610f54a39ff6c81f323159429802dbfa5278e` | 688 | 106.264311 |
| piece_69862af5096ee3fa | `d30b0330aa434fe9c56b4bd130b6799022d1029c2a43cf7e7288c6c7094750ab` | `511ca469e22ba598c646600993c7b07080602c67b88c6357c0117e0835f399e1` | 2630 | 297.443061 |
| piece_ab42b317dcb38fc5 | `7189382c407f57b9e7ed2f197d0d50df31ff305b42957e82ce081934b7ccf369` | `f4f9024bc29e3c68f1241d9109f8ff0d4ec2b13ee74e87ca24dde288b0b6dd2e` | 1888 | 138.151141 |
| piece_b45364ab6817194a | `08bccbe06cf4193cd38805d9216e4ad1a76b560c2bc1fa6a4677f80f074e6e2a` | `1af2661bdd816b783d8bc936620999ccf601cdba6315ef368527b07a8f613e65` | 4595 | 384.661098 |
| piece_18921bc81d708d16 | `9ba3ab32cc956e3167b9a18d498867b3e147aab97899c9a44a5687ab15e103c2` | `64f3d683e862bbc7f41fb255c1996bfa455052d77601e9ff7e76600799aaf748` | 1785 | 178.037848 |
| piece_db97fbaed2036f5b | `76cd56e95a9a9a2174cc0c338be3f96ce806cbaa2f4b63ea6ceb82ddfa6b3ec8` | `7accf9278ce802e5d84f38ead3daacf90587b07fd04e8b66b8a0102a97f6b15d` | 2634 | 129.635811 |
| piece_e7ad71b998a8d477 | `661e642c10966bc6577e007057e7222e6c914f97ca76aad0bf0a805cf4111039` | `1bc838580130004df5823589d401f9fca21025bca5958967029321a81073019a` | 1497 | 180.299665 |
| piece_e38184a2e9753bdc | `b09e50b4320a937f45a663f282ef75a1a81127510f57bc5c5fce251b380b0d5d` | `4a6a1f01e879de51759c9a98d7ccc78b28a4c3f43c1d4716298d08cc5a791856` | 2093 | 163.639811 |
| piece_0a888a7d5fbd06e6 | `9bd61f3ae04fad295bced239d1dbe5309a3467d6c49058b60778e227d4993cb4` | `08440b7edac432002d4e30d0b69dd11efee7cb7ba4c258bbf7101499fa3b82f8` | 2648 | 271.309671 |
| piece_901ec1910fe9fa9e | `2213a5fd6f0bb6cb2af5b7d31c2ce510f6af899d878da97c081c1141a1917238` | `3373a58863542e9e262fa7ba6e71598a22347fb2db0dce848638b6f7d8a87798` | 3105 | 164.170250 |
| piece_4b31472ad4376d9e | `7455a742cfce85f0c3afe1ac231e233595afc3c970641b66fef6f44c2988de93` | `1d38e8ec4b1b865d64b83f556b8f5546bbac30382afb95675b7faac160645339` | 4442 | 265.991838 |
| piece_7195bbce81550519 | `d30d1b8128fb0a03395ed351a4f8c494b5fdfa191faef451bc84f4a244fb6128` | `2fbd11c5fd8930bd47d5358f9e8f8d0994d3a465f06645204bb6a61ec171a3b0` | 10177 | 522.638386 |
| piece_de3f82957f1b3532 | `24b13ec5984c594f1d363901a6441f1c7ac5e8496512bc8aac6944a75ae66665` | `921ec36ea43a03a29a1d54a764512ae4d6eebefd3f93b55bddf36a7b4af5b6e6` | 1816 | 160.961664 |
| piece_759b324c1cceac54 | `53f5cd7475eff99e92a5c410f8441944b45909572d93091e50d2d714244d51ee` | `7881c147530f1d421c63e9bf5113c081ee46bdbecec19fe42308c554e84d4e37` | 2419 | 299.830300 |
| piece_daefdda4e1923cc6 | `b9fcf1c9bfc9275d82c140ed8460751fb8ede135a657b84f0c80b556d6c87cb1` | `97c300391b541b8f0f41a1fec2d3f79509f17f2bb2bfe00b72473b7be3cae31c` | 2620 | 278.344713 |
| piece_3056e93b3e51b646 | `a2e159000a52cdb7dc56c4935e56b99a269c1315b69844373c14dda96c438489` | `849f64935dc844e8419cdca339ca9be4874e10230820a94f163708108346c21f` | 729 | 135.772966 |
| piece_555ff6cd3a13efb9 | `8ea71988ab7601c75787b01dc805a19db5d27066613f52f000b2f62d86d2bb99` | `3bfa454309844c2eea155224b7b5c80f1ec1769e87401dea830445afd80b0090` | 1387 | 192.351623 |

## 8. Fixed validation Original PT baseline

- Official strict base-2 JS distance: **0.161498978463802** (lower is better).
- Official Intersection: **0.821025730937020** (higher is better).
- Both human and candidate paths use MIDI load plus official midi_to_ids; candidate MIDI is the frozen file after dump/reload.

## 9. Distribution diagnostics

- Canonical Original PT steady mass P(0000)+P(1111): `0.952140388194629`.
- Canonical Original PT transition-containing mass: `0.047859611805371`.
- Full human/candidate 16-pattern distribution: `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_joint16_distribution.csv`.

## 10. Existing code/cache reusable for later training

- Shared cache `/workspace/project/analysis/stage2_binary_v0/train_setup_v0/shared_cache`: cache ID `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`, train 2,062 performances / 9,369,095 notes / 35,573 512/256 windows.
- Existing threshold-64 target generation, pedal masking, pretrained encoder initialization, independent 4x2 and joint-16 heads, AdamW LR configuration, CC64-only transplant, and strict non-CC64 assertion are path-compatible.
- Strict future validation must use the pinned MIDI-roundtrip evaluator semantics, not the legacy direct-token headline metric. No training cache was rebuilt and no Stage 2 training was started.

## 11. ASAP test isolation

ASAP test human MIDI loads, canonical test-bank metric reads, prior test-result model choice, and test evaluation were not performed by this runner. Test remains deferred until validation-only model/checkpoint selection is complete.
