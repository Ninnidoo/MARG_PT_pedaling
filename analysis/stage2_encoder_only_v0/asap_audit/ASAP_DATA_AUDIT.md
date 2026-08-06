# ASAP v1.1 Data Audit for Stage 2

## 1. Dataset location and version

- Host path: `/public/intern_2026_summer_public_dataset/ilkyun_data/ASAP/asap-dataset-v1.1`
- Container path: `/workspace/public/ASAP/asap-dataset-v1.1`
- Git tag: `v1.1`
- Git commit: `fad8d1e8078d0ae47ad2f280b5d022bd2de24784`
- Metadata path: `/workspace/public/ASAP/asap-dataset-v1.1/metadata.csv`

## 2. Dataset summary

| Metric | Count |
| --- | ---: |
| Metadata rows | 1,067 |
| Unique performances | 1,067 |
| Unique pieces | 222 |
| Composers | 16 |
| Existing / missing files | 1067 / 0 |
| Tokenizer successes / failures | 1067 / 0 |
| Normalized notes | 3,604,118 |
| Tokens | 28,832,944 |
| Raw CC64 events | 4,509,992 |

Raw notes and raw CC64 events count non-drum instruments, matching the original normalization input. `piece_id` is the first 16 hexadecimal characters of SHA-256 over UTF-8 `(composer + NUL + title)`.

## 3. Pedal statistics

| Value class | Count | Ratio |
| --- | ---: | ---: |
| Zero (0) | 3,708,796 | 25.726100% |
| Full (127) | 5,281,862 | 36.637688% |
| Intermediate (1–126) | 5,425,814 | 37.636212% |

Total pedal tokens: 14,416,472. Pedal1–4 intermediate ratios: Pedal1 37.898093%; Pedal2 37.808945%; Pedal3 37.465421%; Pedal4 37.372389%.

- Performances with no raw CC64 events: 13 (1.22%)
- Performances with at least one raw CC64 event: 1,054 (98.78%)
- Performances containing intermediate pedal values: 1,048 (98.22%)

## 4. Sequence-length statistics

- Normalized note count: min 414; median 2,912; mean 3,377.805; p90 6,355; p95 7,317.900; max 17,080
- Token length: min 3,312; median 23,296; mean 27,022.440; p90 50,840; p95 58,543.200; max 136,640
- Raw CC64 event count: min 0; median 3,478; mean 4,226.797; p90 8,583.200; p95 10,232.100; max 31,133
- Requiring 4096-token windowing: 1,055 (98.88%)
- Above 8192 tokens: 947 (88.75%)

### Top 10 longest performances by token count

| Rank | Metadata index | Composer | Title | Performance path | Tokens |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 861 | Liszt | Sonata | `Liszt/Sonata/GiacomelliN11M.mid` | 136,640 |
| 2 | 860 | Liszt | Sonata | `Liszt/Sonata/Gasanov06M.mid` | 136,128 |
| 3 | 859 | Liszt | Sonata | `Liszt/Sonata/Dvorkine03.mid` | 132,048 |
| 4 | 864 | Liszt | Sonata | `Liszt/Sonata/Zuber07M.mid` | 130,784 |
| 5 | 863 | Liszt | Sonata | `Liszt/Sonata/Yeletskiy05M.mid` | 130,384 |
| 6 | 858 | Liszt | Sonata | `Liszt/Sonata/Dulu07M.mid` | 128,576 |
| 7 | 1023 | Schubert | Wanderer_fantasie | `Schubert/Wanderer_fantasie/Kolessova02.mid` | 127,832 |
| 8 | 862 | Liszt | Sonata | `Liszt/Sonata/Huang01.mid` | 127,824 |
| 9 | 1025 | Schubert | Wanderer_fantasie | `Schubert/Wanderer_fantasie/SunY10M.mid` | 126,720 |
| 10 | 1024 | Schubert | Wanderer_fantasie | `Schubert/Wanderer_fantasie/Larionova05M.mid` | 125,864 |

## 5. Validation results

| Check | Result | Detail |
| --- | --- | --- |
| Metadata path resolution | PASS | All metadata performance paths resolved beneath the verified ASAP root. |
| File existence | PASS | 1067/1067 files exist; 0 missing and 0 zero-byte. |
| Exact eight-token grammar | PASS | Every successful tokenization had length divisible by 8. |
| Token vocabulary ranges | PASS | Every successful tokenization was checked against all eight pinned ranges. |
| Note/token-count consistency | PASS | Every successful tokenization satisfied tokens = normalized notes × 8 and pedal tokens = normalized notes × 4. |
| Duplicate path inspection | PASS | 1067 unique paths; 0 duplicate metadata references. |

## 6. Failures and exclusions

There were no tokenizer failures. No successful performances were excluded, including performances with no raw CC64 events; they are reported separately above for later policy discussion.

## 7. Split implications

- Unique `(composer, title)` pieces: 222.
- Performances per piece: min 1; median 3; mean 4.806; p90 11; p95 14.950; max 29.
- A deterministic piece-level split using `(composer, title)` is feasible; all metadata rows have a deterministic piece identity and no split has been created.

## 8. Recommendation for the next task

First define an exclusion policy, because successful performances with no raw CC64 events need an explicit inclusion decision before Stage 2.
