# Stage 2 Encoder-only Endpoint-aware 5-class Pedal Training v0

## 1. 연구 질문과 가설

동일한 official pretrained PT encoder, ASAP piece-wise split, optimizer, seed, windowing, and training budget에서 Pedal1–4 output만 five coarse classes로 바꾸었을 때 128-class posterior fragmentation과 endpoint collapse가 완화되는지를 validation-only로 검증했다.

## 2. 5-class 정의

| ID | Class | Raw target range |
| ---: | --- | --- |
| 0 | ZERO | 0–0 |
| 1 | LOW | 1–63 |
| 2 | MID | 64–95 |
| 3 | HIGH | 96–126 |
| 4 | FULL | 127–127 |

## 3. Canonical representative 정의

고정 representative는 `[0, 32, 80, 111, 127]`이며 policy는 `canonical_interval_midpoint`이다. Train/validation 분포로 재추정하지 않았다.

## 4. LOW를 1–63으로 합친 이유

CC64 sustain threshold 미만의 nonzero physical pedal positions를 하나의 OFF-region depth class로 표현하여 flat 128-way fragmentation을 줄이는 고정 가설이다.

## 5. ZERO와 LOW를 분리한 이유

완전 release endpoint 0과 subthreshold but physically nonzero 1–63을 구분하여 release collapse를 직접 측정한다.

## 6. 데이터 및 piece-wise split

```json
{
  "asap_root": "/workspace/public/ASAP/asap-dataset-v1.1",
  "split_csv": "/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv",
  "split_csv_sha256": "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b",
  "stride_notes": 256,
  "test_rows_passed_to_pipeline": 0,
  "train_performance_count": 892,
  "train_piece_count": 180,
  "validation_performance_count": 71,
  "validation_piece_count": 19,
  "window_notes": 512
}
```

## 7. 모델 architecture

official PT encoder + four independent Linear(768,5) heads. Encoder는 unfrozen이고 four prediction heads는 independent Linear(768,5)이다.

## 8. Loss와 decoding

Loss는 valid note/slot 전체의 unweighted five-class cross-entropy이다. Final reconstruction은 raw logits window-average → one argmax → canonical representative 순서이며 smoothing/calibration은 없다.

## 9. Implementation tests

```json
{
  "asap_test_midi_accessed": false,
  "errors": [],
  "excluded_module": "test_stage2_dataset.py",
  "excluded_reason": "contains a real ASAP smoke test that opens the test split",
  "failures": [],
  "modules": [
    "test_stage2_five_class.py",
    "test_stage2_five_class_runtime.py",
    "test_stage2_ordinal.py",
    "test_stage2_coarse_to_fine.py",
    "test_stage2_calibration.py",
    "test_stage2_posterior.py",
    "test_stage2_model.py",
    "test_stage2_training.py",
    "test_stage2_train.py",
    "test_stage2_evaluate.py",
    "test_stage2_five_class_audit.py"
  ],
  "passed": true,
  "passed_tests": 107,
  "post_patch_runtime_tests": {
    "errors": 0,
    "failures": 0,
    "passed_tests": 11,
    "skipped": 0,
    "total_tests": 11
  },
  "reused_from_interrupted_prelaunch": true,
  "skipped": [],
  "total_tests": 107
}
```

## 10. Pedal-rich GPU overfit result

```json
{
  "active_encoder_changed": true,
  "active_encoder_parameter": "embeddings.word_embeddings.weight",
  "all_five_classes_present": true,
  "all_losses_finite": true,
  "all_metrics_finite": true,
  "encoder_gradient_finite_nonzero": true,
  "every_head_parameter_group_changed": true,
  "every_head_received_finite_nonzero_gradient": true,
  "final": {
    "all_five_classes_predicted": true,
    "ce_loss": 0.02398877590894699,
    "decoded_mae_gap": 0.017822265625,
    "exact_note_accuracy": 0.99658203125,
    "per_class_recall": {
      "FULL": 1.0,
      "HIGH": 0.9951397326852977,
      "LOW": 0.9995196926032661,
      "MID": 1.0,
      "ZERO": 0.9880952380952381
    },
    "predicted_class_counts": [
      83,
      4162,
      2324,
      819,
      804
    ],
    "quantization_oracle_mae": 9.93798828125,
    "representative_decoded_mae": 9.955810546875,
    "target_class_counts": [
      84,
      4164,
      2317,
      823,
      804
    ],
    "token_accuracy": 0.9991455078125
  },
  "gpu": {
    "name": "NVIDIA GeForce RTX 2080 Ti",
    "uuid": "GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b"
  },
  "head_groups_changed": [
    true,
    true,
    true,
    true
  ],
  "initial": {
    "all_five_classes_predicted": true,
    "ce_loss": 1.9272019863128662,
    "decoded_mae_gap": 36.4683837890625,
    "exact_note_accuracy": 0.0009765625,
    "per_class_recall": {
      "FULL": 0.01616915422885572,
      "HIGH": 0.267314702308627,
      "LOW": 0.3712776176753122,
      "MID": 0.156236512731981,
      "ZERO": 0.21428571428571427
    },
    "predicted_class_counts": [
      1639,
      3054,
      1101,
      2275,
      123
    ],
    "quantization_oracle_mae": 9.93798828125,
    "representative_decoded_mae": 46.4063720703125,
    "target_class_counts": [
      84,
      4164,
      2317,
      823,
      804
    ],
    "token_accuracy": 0.2635498046875
  },
  "oom": false,
  "overfit_optimizer": {
    "encoder_lr": 0.0001,
    "head_lr": 0.001,
    "max_grad_norm": 1.0,
    "name": "AdamW",
    "weight_decay": 0.0
  },
  "passed": true,
  "peak_gpu_memory_bytes": 2497611264,
  "performance_path": "Schubert/Piano_Sonatas/664-3/Lin07.mid",
  "source_split": "train",
  "steps": 25,
  "subset_canonical_quantization_oracle_mae": 9.93798828125,
  "target_class_counts": [
    84,
    4164,
    2317,
    823,
    804
  ],
  "target_class_distribution": [
    0.01025390625,
    0.50830078125,
    0.2828369140625,
    0.1004638671875,
    0.09814453125
  ],
  "test_split_accessed": false,
  "window_starts": [
    0,
    256,
    512,
    768
  ]
}
```

## 11. Training summary

```json
{
  "best_epoch": 3,
  "best_validation_loss": 1.1156102244177202,
  "completed_epoch": 7,
  "early_stopped": true,
  "encoder_initial_sha256": "4bfcd1c68006b148799675b126495e9af750215ae5ed77d63621f6139da2ebe4",
  "four_head_initial_sha256": [
    "c6b4967b55cd1583fa5e67476af8a581c6020bf042d775363d0672cd9e7a26ef",
    "f27b3db56320af4936e568c11a3e491bb34de4a7f2ca0c024b6effb3181559af",
    "20a393e0ea28045645ece48dea2b61c214d070c25643a2fd2a332c641dbb853a",
    "627428a05612b622389b4e9ee35a7e2cb55fa4f57e98ae6d04ea29a1ec5f69dc"
  ],
  "global_optimizer_step": 4914,
  "initial_parameter_sha256": "513aa5d04790608d22fcab85f9183b283982cd567fd6aef9c077b7f3ab47dda0",
  "runtime_seconds": 1835.1775304190814,
  "stop_reason": "early_stopping",
  "train_performance_count": 892,
  "train_piece_count": 180,
  "training_end_time_kst": "2026-08-04T17:16:28.232235+09:00",
  "training_end_time_utc": "2026-08-04T08:16:28.232213+00:00",
  "training_start_time_kst": "2026-08-04T16:44:46.799645+09:00",
  "training_start_time_utc": "2026-08-04T07:44:46.799634+00:00",
  "validation_performance_count": 71,
  "validation_piece_count": 19
}
```

## 12. Best epoch와 validation loss

- Best epoch: `3`
- Best validation five-class CE: `1.1156102244177202`
- Stop reason: `early_stopping`

## 13. 5-class classification metrics

```json
{
  "five_class_exact_note_accuracy": 0.5053957341297793,
  "five_class_macro_f1": 0.34163966028264364,
  "five_class_micro_f1": 0.5800229283480319,
  "five_class_token_accuracy": 0.5800229283480319,
  "five_class_weighted_f1": 0.5201120408360368,
  "predicted_full_ratio": 0.5716097038685864,
  "predicted_high_ratio": 0.0002800005635231467,
  "predicted_low_ratio": 0.03552925389535375,
  "predicted_mid_ratio": 0.0715427854949142,
  "predicted_zero_ratio": 0.3210382561776225
}
```

## 14. Class별 precision/recall/F1

```json
{
  "all_five_classes_predicted": true,
  "confusion_matrix": [
    [
      191325,
      11342,
      9794,
      27,
      71922
    ],
    [
      80506,
      11677,
      13701,
      21,
      31994
    ],
    [
      43672,
      8884,
      25838,
      122,
      59675
    ],
    [
      12446,
      2262,
      8820,
      53,
      55747
    ],
    [
      36658,
      6186,
      23099,
      95,
      429846
    ]
  ],
  "exact_note_accuracy": 0.5053957341297793,
  "f1": [
    0.5895839400200611,
    0.13101823281907435,
    0.23548711966205343,
    0.0013308891846420408,
    0.7507781197273873
  ],
  "full_recall": 0.8668277258391075,
  "high_recall": 0.000668112141992739,
  "high_to_mid_count": 8820,
  "high_to_mid_rate": 0.11118394513916902,
  "low_recall": 0.08467791644609461,
  "low_to_mid_count": 13701,
  "low_to_mid_rate": 0.09935532527429496,
  "low_to_zero_count": 80506,
  "low_to_zero_rate": 0.583804088499554,
  "macro_f1": 0.34163966028264364,
  "macro_precision": 0.3921852966301837,
  "macro_recall": 0.3623710541563716,
  "micro_f1": 0.5800229283480319,
  "micro_precision": 0.5800229283480319,
  "micro_recall": 0.5800229283480319,
  "mid_high_full_aggregate_recall": 0.8456580642357826,
  "mid_recall": 0.18697310244516646,
  "mid_to_high_count": 122,
  "mid_to_high_rate": 0.0008828360747081937,
  "nonendpoint_endpoint_collapse_ratio": 0.7991716795435234,
  "precision": [
    0.5247430795349514,
    0.289385641000223,
    0.31799832619504753,
    0.16666666666666666,
    0.6621327697540297
  ],
  "predicted_count": [
    364607,
    40351,
    81252,
    318,
    649184
  ],
  "predicted_distribution": [
    0.3210382561776225,
    0.03552925389535375,
    0.0715427854949142,
    0.0002800005635231467,
    0.5716097038685864
  ],
  "recall": [
    0.6727084139094969,
    0.08467791644609461,
    0.18697310244516646,
    0.000668112141992739,
    0.8668277258391075
  ],
  "support": [
    284410,
    137899,
    138191,
    79328,
    495884
  ],
  "target_count": 1135712,
  "target_distribution": [
    0.25042440336986843,
    0.12142074751345412,
    0.12167785494914203,
    0.06984869403510749,
    0.4366283001324279
  ],
  "token_accuracy": 0.5800229283480319,
  "valid_note_count": 283928,
  "weighted_f1": 0.5201120408360368,
  "weighted_precision": 0.5059866023992752,
  "weighted_recall": 0.5800229283480319,
  "zero_low_aggregate_recall": 0.6981854518847573,
  "zero_recall": 0.6727084139094969,
  "zero_to_low_count": 11342,
  "zero_to_low_rate": 0.039879047853451005
}
```

## 15. ZERO/LOW confusion analysis

```json
{
  "confusion_matrix": [
    [
      191325,
      11342,
      9794,
      27,
      71922
    ],
    [
      80506,
      11677,
      13701,
      21,
      31994
    ],
    [
      43672,
      8884,
      25838,
      122,
      59675
    ],
    [
      12446,
      2262,
      8820,
      53,
      55747
    ],
    [
      36658,
      6186,
      23099,
      95,
      429846
    ]
  ],
  "high_to_mid_count": 8820,
  "high_to_mid_rate": 0.11118394513916902,
  "low_to_mid_count": 13701,
  "low_to_mid_rate": 0.09935532527429496,
  "low_to_zero_count": 80506,
  "low_to_zero_rate": 0.583804088499554,
  "mid_to_high_count": 122,
  "mid_to_high_rate": 0.0008828360747081937,
  "nonendpoint_endpoint_collapse_ratio": 0.7991716795435234,
  "zero_to_low_count": 11342,
  "zero_to_low_rate": 0.039879047853451005
}
```

## 16. OFF/ON renderer-aware metrics

```json
{
  "on_region_depth_class_accuracy": 0.6388212553073088,
  "on_region_macro_f1": 0.3635890027456217,
  "on_region_per_class_f1": {
    "FULL": 0.8257122879272191,
    "HIGH": 0.0013316917510490212,
    "MID": 0.26372302855859714
  },
  "state": {
    "accuracy": 0.7908210884449579,
    "balanced_accuracy": 0.77192175806027,
    "confusion_matrix": [
      [
        294850,
        127459
      ],
      [
        110108,
        603295
      ]
    ],
    "macro_f1": 0.7741634566595919,
    "off_f1": 0.7128291107949428,
    "off_precision": 0.7281001980452294,
    "off_recall": 0.6981854518847573,
    "off_support": 422309,
    "on_f1": 0.8354978025242409,
    "on_precision": 0.825578785747324,
    "on_recall": 0.8456580642357826,
    "on_support": 713403,
    "predicted_off_count": 404958,
    "predicted_on_count": 730754
  },
  "zero_vs_nonzero": {
    "accuracy": 0.7654625468428615,
    "balanced_accuracy": 0.734579513602683,
    "confusion_matrix": [
      [
        678020,
        173282
      ],
      [
        93085,
        191325
      ]
    ],
    "macro_f1": 0.712701902597846,
    "off_f1": 0.8358198651756311,
    "off_precision": 0.8792836254465993,
    "off_recall": 0.7964506132958692,
    "off_support": 851302,
    "on_f1": 0.5895839400200611,
    "on_precision": 0.5247430795349514,
    "on_recall": 0.6727084139094969,
    "on_support": 284410,
    "predicted_off_count": 771105,
    "predicted_on_count": 364607
  }
}
```

## 17. Transition 및 repedal analysis

```json
{
  "combined_binary_transition": {
    "f1": 0.1271023631327715,
    "precision": 0.10476005440062172,
    "predicted_count": 51470,
    "recall": 0.16155805243445692,
    "true_count": 33375,
    "true_positive": 5392
  },
  "off_to_on_transition": {
    "f1": 0.11204072494167,
    "precision": 0.09231784993009166,
    "predicted_count": 25748,
    "recall": 0.1424803692381466,
    "true_count": 16683,
    "true_positive": 2377
  },
  "on_to_off_transition": {
    "f1": 0.08926297920497948,
    "precision": 0.07359458829017962,
    "predicted_count": 25722,
    "recall": 0.11340762041696621,
    "true_count": 16692,
    "true_positive": 1893
  },
  "short_repedal_like": {
    "definition": ">=64 -> <64 -> >=64 with OFF run <=4 flattened samples; exact recall requires both transition boundaries at the same samples",
    "f1": 0.04344215724837201,
    "precision": 0.030323367907260524,
    "predicted_count": 16390,
    "recall": 0.07656755507625944,
    "sample_based_not_milliseconds": true,
    "true_count": 6491,
    "true_positive": 497
  },
  "steady_position_state_accuracy": 0.7982891606926096,
  "steady_position_state_count": 1102266,
  "transition_direction_accuracy": 0.12794007490636705,
  "transition_timing_sample_offset": {
    "matching": "earliest-feasible one-to-one, within performance and direction",
    "plus_minus_0_samples": {
      "combined": {
        "f1": 0.10065413400907537,
        "matched_count": 4270,
        "maximum_absolute_offset": 0,
        "mean_absolute_offset": 0.0,
        "mean_signed_offset": 0.0,
        "median_absolute_offset": 0.0,
        "precision": 0.08296094812512143,
        "predicted_count": 51470,
        "recall": 0.12794007490636705,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.11204072494167,
        "matched_count": 2377,
        "precision": 0.09231784993009166,
        "predicted_count": 25748,
        "recall": 0.1424803692381466,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.08926297920497948,
        "matched_count": 1893,
        "precision": 0.07359458829017962,
        "predicted_count": 25722,
        "recall": 0.11340762041696621,
        "true_count": 16692
      }
    },
    "plus_minus_1_samples": {
      "combined": {
        "f1": 0.21370734869467853,
        "matched_count": 9066,
        "maximum_absolute_offset": 1,
        "mean_absolute_offset": 0.529009485991617,
        "mean_signed_offset": -0.02316346790205162,
        "median_absolute_offset": 1.0,
        "precision": 0.17614144161647563,
        "predicted_count": 51470,
        "recall": 0.27164044943820226,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.2433126723386204,
        "matched_count": 5162,
        "precision": 0.20048159080316919,
        "predicted_count": 25748,
        "recall": 0.30941677156386743,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.18409015890979397,
        "matched_count": 3904,
        "precision": 0.1517766892154576,
        "predicted_count": 25722,
        "recall": 0.23388449556673857,
        "true_count": 16692
      }
    },
    "plus_minus_2_samples": {
      "combined": {
        "f1": 0.27402911190995344,
        "matched_count": 11625,
        "maximum_absolute_offset": 2,
        "mean_absolute_offset": 0.8966881720430108,
        "mean_signed_offset": -0.09462365591397849,
        "median_absolute_offset": 1.0,
        "precision": 0.2258597241111327,
        "predicted_count": 51470,
        "recall": 0.34831460674157305,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.30529565647757534,
        "matched_count": 6477,
        "precision": 0.25155351871990056,
        "predicted_count": 25748,
        "recall": 0.3882395252652401,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.24275003536568115,
        "matched_count": 5148,
        "precision": 0.2001399580125962,
        "predicted_count": 25722,
        "recall": 0.308411214953271,
        "true_count": 16692
      }
    },
    "plus_minus_4_samples": {
      "combined": {
        "f1": 0.33963109199127817,
        "matched_count": 14408,
        "maximum_absolute_offset": 4,
        "mean_absolute_offset": 1.6474181010549696,
        "mean_signed_offset": -0.43684064408661855,
        "median_absolute_offset": 1.0,
        "precision": 0.27993005634350104,
        "predicted_count": 51470,
        "recall": 0.4317003745318352,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.36742004666399575,
        "matched_count": 7795,
        "precision": 0.3027419605406245,
        "predicted_count": 25748,
        "recall": 0.4672421027393155,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.31183099919837787,
        "matched_count": 6613,
        "precision": 0.257095093694114,
        "predicted_count": 25722,
        "recall": 0.39617780972921157,
        "true_count": 16692
      }
    }
  }
}
```

Short repedal-like metric은 `>=64 → <64 → >=64`, low run ≤4 flattened Pedal samples인 sample-based diagnostic이며 millisecond timing metric이 아니다.

## 18. Canonical decoded-value metrics

```json
{
  "decoded_transition_direction_accuracy": 0.11917995255695923,
  "decoded_transition_f1": 0.21085524775321318,
  "decoded_transition_precision": 0.275538247566064,
  "decoded_transition_recall": 0.1707673636012578,
  "delta_mae": 7.948861479992357,
  "error_quantiles": {
    "median": 0.0,
    "q01": 0.0,
    "q05": 0.0,
    "q25": 0.0,
    "q75": 47.0,
    "q95": 127.0,
    "q99": 127.0
  },
  "exact_value_transition_accuracy": 0.0950446847244442,
  "maximum_absolute_error": 127,
  "median_performance_mae": 28.6976899485556,
  "on_region_mae": 22.016125527927414,
  "overall_micro_mae": 28.764686822011214,
  "per_slot_mae": {
    "Pedal1": 28.825871347665604,
    "Pedal2": 28.828896762559523,
    "Pedal3": 28.699275872756473,
    "Pedal4": 28.704703305063255
  },
  "performance_count": 71,
  "performance_macro_mae": 27.46777780720563,
  "steady_position_decoded_accuracy": 0.6140335646687697,
  "subthreshold_region_mae": 48.430191662013506,
  "target_count": 1135712,
  "tolerance_accuracy": {
    "plus_minus_10": 0.5834709856019836,
    "plus_minus_20": 0.6166211152123073,
    "plus_minus_5": 0.5699288199825308
  },
  "zero_target_mae": 36.15750149432158
}
```

## 19. Validation quantization oracle와 model gap

```json
{
  "decoded_transition_direction_accuracy": 0.521328680973134,
  "decoded_transition_f1": 0.6853596957623393,
  "decoded_transition_precision": 1.0,
  "decoded_transition_recall": 0.521328680973134,
  "delta_mae": 1.434443631394076,
  "error_quantiles": {
    "median": 0.0,
    "q01": 0.0,
    "q05": 0.0,
    "q25": 0.0,
    "q75": 4.0,
    "q95": 19.0,
    "q99": 29.0
  },
  "exact_value_transition_accuracy": 0.18465548629116788,
  "maximum_absolute_error": 31,
  "median_performance_mae": 3.4748825352394284,
  "on_region_mae": 2.5971043014957886,
  "overall_micro_mae": 3.635270209348849,
  "per_slot_mae": {
    "Pedal1": 3.654623707418782,
    "Pedal2": 3.6714624834465077,
    "Pedal3": 3.616568284917303,
    "Pedal4": 3.598426361612803
  },
  "performance_count": 71,
  "performance_macro_mae": 4.171420050076859,
  "steady_position_decoded_accuracy": 0.7683290851735016,
  "subthreshold_region_mae": 16.503658474680744,
  "target_count": 1135712,
  "tolerance_accuracy": {
    "plus_minus_10": 0.8387918768138402,
    "plus_minus_20": 0.9544911033783212,
    "plus_minus_5": 0.7662065734975064
  },
  "zero_target_mae": 0.0
}
```

Model row oracle/gap:

```json
{
  "canonical_oracle_median_performance_mae": 3.4748825352394284,
  "canonical_oracle_on_region_mae": 2.5971043014957886,
  "canonical_oracle_overall_mae": 3.635270209348849,
  "canonical_oracle_pedal1_mae": 3.654623707418782,
  "canonical_oracle_pedal2_mae": 3.6714624834465077,
  "canonical_oracle_pedal3_mae": 3.616568284917303,
  "canonical_oracle_pedal4_mae": 3.598426361612803,
  "canonical_oracle_performance_macro_mae": 4.171420050076859,
  "canonical_oracle_subthreshold_mae": 16.503658474680744,
  "model_minus_oracle_on_region_mae": 19.419021226431624,
  "model_minus_oracle_overall_mae": 25.129416612662364
}
```

Train canonical oracle:

```json
{
  "all_five_classes_present": true,
  "class_boundaries": [
    [
      0,
      0
    ],
    [
      1,
      63
    ],
    [
      64,
      95
    ],
    [
      96,
      126
    ],
    [
      127,
      127
    ]
  ],
  "class_counts": [
    2975658,
    1758841,
    2014383,
    869828,
    4335746
  ],
  "class_distribution": [
    0.24891622002707608,
    0.14712848497664804,
    0.16850478181524947,
    0.07276182203523104,
    0.3626886911457953
  ],
  "class_names": [
    "ZERO",
    "LOW",
    "MID",
    "HIGH",
    "FULL"
  ],
  "error_quantiles": {
    "median": 0.0,
    "q01": 0.0,
    "q05": 0.0,
    "q25": 0.0,
    "q75": 8.0,
    "q95": 23.0,
    "q99": 30.0
  },
  "maximum_absolute_error": 31,
  "median_performance_mae": 4.156661229409783,
  "note_count": 2988614,
  "on_region_mae": 3.3104990791496403,
  "overall_mae": 4.535356188520833,
  "per_slot_mae": [
    4.563787093281367,
    4.590765819875032,
    4.505037452143368,
    4.481834388783564
  ],
  "performance_count": 892,
  "performance_macro_mae": 4.662331420607985,
  "piece_count": 180,
  "representative_policy": "canonical_interval_midpoint",
  "representatives": [
    0,
    32,
    80,
    111,
    127
  ],
  "source_split": "train",
  "split_csv_sha256": "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b",
  "subthreshold_region_mae": 17.236381799150692,
  "target_count": 11954456,
  "test_split_accessed": false,
  "tolerance_accuracy": {
    "plus_minus_10": 0.8036634205688657,
    "plus_minus_20": 0.9377912303161264,
    "plus_minus_5": 0.7128965968840406
  },
  "zero_target_mae": 0.0
}
```

## 20. 기존 CE argmax 및 posterior-median rebinned comparison

Baseline CE argmax:

```json
{
  "all_five_classes_predicted": false,
  "architecture": "official PT encoder + four independent Linear(768, 128) heads",
  "best_epoch": 2,
  "best_validation_loss": 2.2459586270443803,
  "binary_transition_f1": 0.09522499312138873,
  "binary_transition_precision": 0.0817250928450293,
  "binary_transition_recall": 0.11406741573033707,
  "canonical_decoded_delta_mae": 7.839006340912313,
  "canonical_decoded_error_median": 0.0,
  "canonical_decoded_error_q01": 0.0,
  "canonical_decoded_error_q05": 0.0,
  "canonical_decoded_error_q25": 0.0,
  "canonical_decoded_error_q75": 56.0,
  "canonical_decoded_error_q95": 127.0,
  "canonical_decoded_error_q99": 127.0,
  "canonical_decoded_exact_value_transition_accuracy": 0.10699509019694378,
  "canonical_decoded_maximum_absolute_error": 127,
  "canonical_decoded_median_performance_mae": 30.575388523610282,
  "canonical_decoded_on_region_mae": 22.05786070425832,
  "canonical_decoded_overall_mae": 30.86541922600096,
  "canonical_decoded_pedal1_mae": 30.913770392493873,
  "canonical_decoded_pedal2_mae": 31.08634231213547,
  "canonical_decoded_pedal3_mae": 30.742550928404384,
  "canonical_decoded_pedal4_mae": 30.719013270970105,
  "canonical_decoded_performance_macro_mae": 29.528470064464777,
  "canonical_decoded_steady_position_accuracy": 0.6388986750788643,
  "canonical_decoded_subthreshold_mae": 56.04884009311162,
  "canonical_decoded_tolerance_10": 0.5914166619706405,
  "canonical_decoded_tolerance_20": 0.6117395959539038,
  "canonical_decoded_tolerance_5": 0.5851985362486264,
  "canonical_decoded_transition_direction_accuracy": 0.053621669333039114,
  "canonical_decoded_transition_f1": 0.1160548854639116,
  "canonical_decoded_transition_precision": 0.23867075971921087,
  "canonical_decoded_transition_recall": 0.07666740221768632,
  "canonical_decoded_zero_mae": 40.74752645828206,
  "canonical_oracle_median_performance_mae": 3.4748825352394284,
  "canonical_oracle_on_region_mae": 2.5971043014957886,
  "canonical_oracle_overall_mae": 3.635270209348849,
  "canonical_oracle_pedal1_mae": 3.654623707418782,
  "canonical_oracle_pedal2_mae": 3.6714624834465077,
  "canonical_oracle_pedal3_mae": 3.616568284917303,
  "canonical_oracle_pedal4_mae": 3.598426361612803,
  "canonical_oracle_performance_macro_mae": 4.171420050076859,
  "canonical_oracle_subthreshold_mae": 16.503658474680744,
  "checkpoint": "/workspace/project/analysis/stage2_encoder_only_v0/train_v0/best.pt",
  "decoder": "argmax after overlap-averaged 128-class raw logits",
  "five_class_exact_note_accuracy": 0.5178848158688119,
  "five_class_macro_f1": 0.2613990976117308,
  "five_class_micro_f1": 0.5709739793187005,
  "five_class_token_accuracy": 0.5709739793187005,
  "five_class_weighted_f1": 0.46355345021347366,
  "full_f1": 0.7317241329434482,
  "full_precision": 0.6082195074707615,
  "full_predicted_count": 748585,
  "full_recall": 0.918166345354962,
  "full_support": 495884,
  "high_f1": 0.0,
  "high_precision": 0.0,
  "high_predicted_count": 0,
  "high_recall": 0.0,
  "high_support": 79328,
  "high_to_mid_count": 0,
  "high_to_mid_rate": 0.0,
  "is_new_model": false,
  "is_primary_reference": false,
  "label": "CE 128-class argmax",
  "low_f1": 0.0,
  "low_precision": 0.0,
  "low_predicted_count": 0,
  "low_recall": 0.0,
  "low_support": 137899,
  "low_to_mid_count": 0,
  "low_to_mid_rate": 0.0,
  "low_to_zero_count": 89458,
  "low_to_zero_rate": 0.6487211654906853,
  "mid_f1": 0.0,
  "mid_high_full_aggregate_recall": 0.8535035596990761,
  "mid_precision": 0.0,
  "mid_predicted_count": 0,
  "mid_recall": 0.0,
  "mid_support": 138191,
  "mid_to_high_count": 0,
  "mid_to_high_rate": 0.0,
  "model": "ce_128_argmax",
  "model_minus_oracle_on_region_mae": 19.46075640276253,
  "model_minus_oracle_overall_mae": 27.23014901665211,
  "nonendpoint_endpoint_collapse_ratio": 1.0,
  "off_f1": 0.6983035100983895,
  "off_on_balanced_accuracy": 0.7613598511906651,
  "off_on_state_accuracy": 0.7849771773125581,
  "off_on_state_macro_f1": 0.7656339696774954,
  "off_precision": 0.7300343298194133,
  "off_recall": 0.669216142682254,
  "off_to_on_transition_f1": 0.07226806701675419,
  "off_to_on_transition_precision": 0.06199854121079504,
  "off_to_on_transition_predicted_count": 23307,
  "off_to_on_transition_recall": 0.08661511718515855,
  "off_to_on_transition_true_count": 16683,
  "off_to_on_transition_true_positive": 1445,
  "on_f1": 0.8329644292566013,
  "on_precision": 0.8133905969262007,
  "on_recall": 0.8535035596990761,
  "on_region_3way_accuracy": 0.6382143052384136,
  "on_region_macro_f1": 0.2747489083759965,
  "on_to_off_transition_f1": 0.06745396317053644,
  "on_to_off_transition_precision": 0.05791373088159477,
  "on_to_off_transition_predicted_count": 23276,
  "on_to_off_transition_recall": 0.08075724898154804,
  "on_to_off_transition_true_count": 16692,
  "on_to_off_transition_true_positive": 1348,
  "predicted_full_ratio": 0.6591327730974049,
  "predicted_high_ratio": 0.0,
  "predicted_low_ratio": 0.0,
  "predicted_mid_ratio": 0.0,
  "predicted_off_ratio": 0.340867226902595,
  "predicted_on_ratio": 0.6591327730974049,
  "predicted_zero_ratio": 0.340867226902595,
  "short_repedal_f1": 0.026820636990128516,
  "short_repedal_precision": 0.01921921921921922,
  "short_repedal_predicted_count": 14985,
  "short_repedal_recall": 0.04436912648282237,
  "short_repedal_true_count": 6491,
  "short_repedal_true_positive": 288,
  "steady_position_state_accuracy": 0.7930626545679537,
  "target_full_ratio": 0.4366283001324279,
  "target_high_ratio": 0.06984869403510749,
  "target_low_ratio": 0.12142074751345412,
  "target_mid_ratio": 0.12167785494914203,
  "target_zero_ratio": 0.25042440336986843,
  "transition_direction_accuracy": 0.08368539325842697,
  "zero_f1": 0.5752713551152059,
  "zero_low_aggregate_recall": 0.669216142682254,
  "zero_precision": 0.4989525401224921,
  "zero_predicted_count": 387127,
  "zero_recall": 0.679153334974157,
  "zero_support": 284410,
  "zero_to_low_count": 0,
  "zero_to_low_rate": 0.0,
  "zero_vs_nonzero_accuracy": 0.7488615071426559
}
```

Primary reference:

```json
{
  "all_five_classes_predicted": true,
  "architecture": "official PT encoder + four independent Linear(768, 128) heads",
  "best_epoch": 2,
  "best_validation_loss": 2.2459586270443803,
  "binary_transition_f1": 0.07247864196161101,
  "binary_transition_precision": 0.06746882181311162,
  "binary_transition_recall": 0.07829213483146068,
  "canonical_decoded_delta_mae": 7.225409262258055,
  "canonical_decoded_error_median": 11.0,
  "canonical_decoded_error_q01": 0.0,
  "canonical_decoded_error_q05": 0.0,
  "canonical_decoded_error_q25": 0.0,
  "canonical_decoded_error_q75": 47.0,
  "canonical_decoded_error_q95": 127.0,
  "canonical_decoded_error_q99": 127.0,
  "canonical_decoded_exact_value_transition_accuracy": 0.05835907761902135,
  "canonical_decoded_maximum_absolute_error": 127,
  "canonical_decoded_median_performance_mae": 27.9088976459608,
  "canonical_decoded_on_region_mae": 19.22786559630391,
  "canonical_decoded_overall_mae": 28.44264919275309,
  "canonical_decoded_pedal1_mae": 28.454491983883237,
  "canonical_decoded_pedal2_mae": 28.56381547434561,
  "canonical_decoded_pedal3_mae": 28.319260516750724,
  "canonical_decoded_pedal4_mae": 28.433028796032797,
  "canonical_decoded_performance_macro_mae": 26.642831236397758,
  "canonical_decoded_steady_position_accuracy": 0.5059240378548896,
  "canonical_decoded_subthreshold_mae": 42.37665972922211,
  "canonical_decoded_tolerance_10": 0.4969023837029106,
  "canonical_decoded_tolerance_20": 0.5845751387675748,
  "canonical_decoded_tolerance_5": 0.4765336634639768,
  "canonical_decoded_transition_direction_accuracy": 0.1462597230650411,
  "canonical_decoded_transition_f1": 0.2020910663272247,
  "canonical_decoded_transition_precision": 0.19054395758767698,
  "canonical_decoded_transition_recall": 0.21512798587742044,
  "canonical_decoded_zero_mae": 44.80061179283429,
  "canonical_oracle_median_performance_mae": 3.4748825352394284,
  "canonical_oracle_on_region_mae": 2.5971043014957886,
  "canonical_oracle_overall_mae": 3.635270209348849,
  "canonical_oracle_pedal1_mae": 3.654623707418782,
  "canonical_oracle_pedal2_mae": 3.6714624834465077,
  "canonical_oracle_pedal3_mae": 3.616568284917303,
  "canonical_oracle_pedal4_mae": 3.598426361612803,
  "canonical_oracle_performance_macro_mae": 4.171420050076859,
  "canonical_oracle_subthreshold_mae": 16.503658474680744,
  "checkpoint": "/workspace/project/analysis/stage2_encoder_only_v0/train_v0/best.pt",
  "decoder": "smallest 128-class posterior cumulative value reaching 0.5",
  "five_class_exact_note_accuracy": 0.428524132878758,
  "five_class_macro_f1": 0.37138145748758156,
  "five_class_micro_f1": 0.5156456918655434,
  "five_class_token_accuracy": 0.5156456918655434,
  "five_class_weighted_f1": 0.5160466645764915,
  "full_f1": 0.7345342812857482,
  "full_precision": 0.7117773910411622,
  "full_predicted_count": 528640,
  "full_recall": 0.7587943954634552,
  "full_support": 495884,
  "high_f1": 0.1429939776943487,
  "high_precision": 0.12692270106518128,
  "high_predicted_count": 102330,
  "high_recall": 0.16372529245663575,
  "high_support": 79328,
  "high_to_mid_count": 12744,
  "high_to_mid_rate": 0.16064945542557482,
  "is_new_model": false,
  "is_primary_reference": true,
  "label": "CE 128-class posterior median",
  "low_f1": 0.21108522722845943,
  "low_precision": 0.228049796722305,
  "low_predicted_count": 118803,
  "low_recall": 0.1964698801296601,
  "low_support": 137899,
  "low_to_mid_count": 27889,
  "low_to_mid_rate": 0.20224222075577053,
  "low_to_zero_count": 53220,
  "low_to_zero_rate": 0.38593463331858824,
  "mid_f1": 0.253904869007175,
  "mid_high_full_aggregate_recall": 0.881333552003566,
  "mid_precision": 0.2374220766954222,
  "mid_predicted_count": 158810,
  "mid_recall": 0.27284700161370856,
  "mid_support": 138191,
  "mid_to_high_count": 21889,
  "mid_to_high_rate": 0.15839671179743978,
  "model": "ce_128_posterior_median",
  "model_minus_oracle_on_region_mae": 16.63076129480812,
  "model_minus_oracle_overall_mae": 24.80737898340424,
  "nonendpoint_endpoint_collapse_ratio": 0.5100979691518156,
  "off_f1": 0.6801902007312809,
  "off_on_balanced_accuracy": 0.7500078035432278,
  "off_on_state_accuracy": 0.7836678665013666,
  "off_on_state_macro_f1": 0.7583715178078281,
  "off_precision": 0.7552784940392909,
  "off_recall": 0.6186820550828895,
  "off_to_on_transition_f1": 0.0524053791764869,
  "off_to_on_transition_precision": 0.048756578268496546,
  "off_to_on_transition_predicted_count": 19382,
  "off_to_on_transition_recall": 0.05664448840136666,
  "off_to_on_transition_true_count": 16683,
  "off_to_on_transition_true_positive": 945,
  "on_f1": 0.8365528348843754,
  "on_precision": 0.7961027121476867,
  "on_recall": 0.881333552003566,
  "on_region_3way_accuracy": 0.5984934181661697,
  "on_region_macro_f1": 0.42663067949599137,
  "on_to_off_transition_f1": 0.05133327783789783,
  "on_to_off_transition_precision": 0.047811030133870884,
  "on_to_off_transition_predicted_count": 19347,
  "on_to_off_transition_recall": 0.05541576803259046,
  "on_to_off_transition_true_count": 16692,
  "on_to_off_transition_true_positive": 925,
  "predicted_full_ratio": 0.4654701191851455,
  "predicted_high_ratio": 0.09010206812994843,
  "predicted_low_ratio": 0.10460662562339748,
  "predicted_mid_ratio": 0.13983298582739287,
  "predicted_off_ratio": 0.30459482685751316,
  "predicted_on_ratio": 0.6954051731424868,
  "predicted_zero_ratio": 0.1999882012341157,
  "short_repedal_f1": 0.010627338285528384,
  "short_repedal_precision": 0.00819946452476573,
  "short_repedal_predicted_count": 11952,
  "short_repedal_recall": 0.015097827761515945,
  "short_repedal_true_count": 6491,
  "short_repedal_true_positive": 98,
  "steady_position_state_accuracy": 0.792038400894158,
  "target_full_ratio": 0.4366283001324279,
  "target_high_ratio": 0.06984869403510749,
  "target_low_ratio": 0.12142074751345412,
  "target_mid_ratio": 0.12167785494914203,
  "target_zero_ratio": 0.25042440336986843,
  "transition_direction_accuracy": 0.05602996254681648,
  "zero_f1": 0.5143889322221766,
  "zero_low_aggregate_recall": 0.6186820550828895,
  "zero_precision": 0.579252319166641,
  "zero_predicted_count": 227129,
  "zero_recall": 0.4625892197883337,
  "zero_support": 284410,
  "zero_to_low_count": 49397,
  "zero_to_low_rate": 0.17368235997327802,
  "zero_vs_nonzero_accuracy": 0.7812746541376687
}
```

New five-class:

```json
{
  "all_five_classes_predicted": true,
  "architecture": "official PT encoder + four Linear(768,5) heads",
  "best_epoch": 3,
  "best_validation_loss": 1.1156102244177202,
  "binary_transition_f1": 0.1271023631327715,
  "binary_transition_precision": 0.10476005440062172,
  "binary_transition_recall": 0.16155805243445692,
  "canonical_decoded_delta_mae": 7.948861479992357,
  "canonical_decoded_error_median": 0.0,
  "canonical_decoded_error_q01": 0.0,
  "canonical_decoded_error_q05": 0.0,
  "canonical_decoded_error_q25": 0.0,
  "canonical_decoded_error_q75": 47.0,
  "canonical_decoded_error_q95": 127.0,
  "canonical_decoded_error_q99": 127.0,
  "canonical_decoded_exact_value_transition_accuracy": 0.0950446847244442,
  "canonical_decoded_maximum_absolute_error": 127,
  "canonical_decoded_median_performance_mae": 28.6976899485556,
  "canonical_decoded_on_region_mae": 22.016125527927414,
  "canonical_decoded_overall_mae": 28.764686822011214,
  "canonical_decoded_pedal1_mae": 28.825871347665604,
  "canonical_decoded_pedal2_mae": 28.828896762559523,
  "canonical_decoded_pedal3_mae": 28.699275872756473,
  "canonical_decoded_pedal4_mae": 28.704703305063255,
  "canonical_decoded_performance_macro_mae": 27.46777780720563,
  "canonical_decoded_steady_position_accuracy": 0.6140335646687697,
  "canonical_decoded_subthreshold_mae": 48.430191662013506,
  "canonical_decoded_tolerance_10": 0.5834709856019836,
  "canonical_decoded_tolerance_20": 0.6166211152123073,
  "canonical_decoded_tolerance_5": 0.5699288199825308,
  "canonical_decoded_transition_direction_accuracy": 0.11917995255695923,
  "canonical_decoded_transition_f1": 0.21085524775321318,
  "canonical_decoded_transition_precision": 0.275538247566064,
  "canonical_decoded_transition_recall": 0.1707673636012578,
  "canonical_decoded_zero_mae": 36.15750149432158,
  "canonical_oracle_median_performance_mae": 3.4748825352394284,
  "canonical_oracle_on_region_mae": 2.5971043014957886,
  "canonical_oracle_overall_mae": 3.635270209348849,
  "canonical_oracle_pedal1_mae": 3.654623707418782,
  "canonical_oracle_pedal2_mae": 3.6714624834465077,
  "canonical_oracle_pedal3_mae": 3.616568284917303,
  "canonical_oracle_pedal4_mae": 3.598426361612803,
  "canonical_oracle_performance_macro_mae": 4.171420050076859,
  "canonical_oracle_subthreshold_mae": 16.503658474680744,
  "checkpoint": "/workspace/project/analysis/stage2_encoder_only_5class_v0/best.pt",
  "decoder": "argmax after overlap-averaged raw logits",
  "five_class_exact_note_accuracy": 0.5053957341297793,
  "five_class_macro_f1": 0.34163966028264364,
  "five_class_micro_f1": 0.5800229283480319,
  "five_class_token_accuracy": 0.5800229283480319,
  "five_class_weighted_f1": 0.5201120408360368,
  "full_f1": 0.7507781197273873,
  "full_precision": 0.6621327697540297,
  "full_predicted_count": 649184,
  "full_recall": 0.8668277258391075,
  "full_support": 495884,
  "high_f1": 0.0013308891846420408,
  "high_precision": 0.16666666666666666,
  "high_predicted_count": 318,
  "high_recall": 0.000668112141992739,
  "high_support": 79328,
  "high_to_mid_count": 8820,
  "high_to_mid_rate": 0.11118394513916902,
  "is_new_model": true,
  "is_primary_reference": false,
  "label": "Endpoint-aware 5-class argmax",
  "low_f1": 0.13101823281907435,
  "low_precision": 0.289385641000223,
  "low_predicted_count": 40351,
  "low_recall": 0.08467791644609461,
  "low_support": 137899,
  "low_to_mid_count": 13701,
  "low_to_mid_rate": 0.09935532527429496,
  "low_to_zero_count": 80506,
  "low_to_zero_rate": 0.583804088499554,
  "mid_f1": 0.23548711966205343,
  "mid_high_full_aggregate_recall": 0.8456580642357826,
  "mid_precision": 0.31799832619504753,
  "mid_predicted_count": 81252,
  "mid_recall": 0.18697310244516646,
  "mid_support": 138191,
  "mid_to_high_count": 122,
  "mid_to_high_rate": 0.0008828360747081937,
  "model": "endpoint_aware_5class_argmax",
  "model_minus_oracle_on_region_mae": 19.419021226431624,
  "model_minus_oracle_overall_mae": 25.129416612662364,
  "nonendpoint_endpoint_collapse_ratio": 0.7991716795435234,
  "off_f1": 0.7128291107949428,
  "off_on_balanced_accuracy": 0.77192175806027,
  "off_on_state_accuracy": 0.7908210884449579,
  "off_on_state_macro_f1": 0.7741634566595919,
  "off_precision": 0.7281001980452294,
  "off_recall": 0.6981854518847573,
  "off_to_on_transition_f1": 0.11204072494167,
  "off_to_on_transition_precision": 0.09231784993009166,
  "off_to_on_transition_predicted_count": 25748,
  "off_to_on_transition_recall": 0.1424803692381466,
  "off_to_on_transition_true_count": 16683,
  "off_to_on_transition_true_positive": 2377,
  "on_f1": 0.8354978025242409,
  "on_precision": 0.825578785747324,
  "on_recall": 0.8456580642357826,
  "on_region_3way_accuracy": 0.6388212553073088,
  "on_region_macro_f1": 0.3635890027456217,
  "on_to_off_transition_f1": 0.08926297920497948,
  "on_to_off_transition_precision": 0.07359458829017962,
  "on_to_off_transition_predicted_count": 25722,
  "on_to_off_transition_recall": 0.11340762041696621,
  "on_to_off_transition_true_count": 16692,
  "on_to_off_transition_true_positive": 1893,
  "predicted_full_ratio": 0.5716097038685864,
  "predicted_high_ratio": 0.0002800005635231467,
  "predicted_low_ratio": 0.03552925389535375,
  "predicted_mid_ratio": 0.0715427854949142,
  "predicted_off_ratio": 0.35656751007297627,
  "predicted_on_ratio": 0.6434324899270237,
  "predicted_zero_ratio": 0.3210382561776225,
  "short_repedal_f1": 0.04344215724837201,
  "short_repedal_precision": 0.030323367907260524,
  "short_repedal_predicted_count": 16390,
  "short_repedal_recall": 0.07656755507625944,
  "short_repedal_true_count": 6491,
  "short_repedal_true_positive": 497,
  "steady_position_state_accuracy": 0.7982891606926096,
  "target_full_ratio": 0.4366283001324279,
  "target_high_ratio": 0.06984869403510749,
  "target_low_ratio": 0.12142074751345412,
  "target_mid_ratio": 0.12167785494914203,
  "target_zero_ratio": 0.25042440336986843,
  "transition_direction_accuracy": 0.12794007490636705,
  "zero_f1": 0.5895839400200611,
  "zero_low_aggregate_recall": 0.6981854518847573,
  "zero_precision": 0.5247430795349514,
  "zero_predicted_count": 364607,
  "zero_recall": 0.6727084139094969,
  "zero_support": 284410,
  "zero_to_low_count": 11342,
  "zero_to_low_rate": 0.039879047853451005,
  "zero_vs_nonzero_accuracy": 0.7654625468428615
}
```

## 21. Ordinal/coarse-to-fine secondary comparison

Ordinal lambda 1.0 posterior median:

```json
{
  "all_five_classes_predicted": true,
  "architecture": "official PT encoder + four independent Linear(768, 128) heads",
  "best_epoch": 2,
  "best_validation_loss": 2.396898931165352,
  "binary_transition_f1": 0.07845664454560412,
  "binary_transition_precision": 0.07232601056702985,
  "binary_transition_recall": 0.08572284644194757,
  "canonical_decoded_delta_mae": 7.254869276470293,
  "canonical_decoded_error_median": 11.0,
  "canonical_decoded_error_q01": 0.0,
  "canonical_decoded_error_q05": 0.0,
  "canonical_decoded_error_q25": 0.0,
  "canonical_decoded_error_q75": 47.0,
  "canonical_decoded_error_q95": 127.0,
  "canonical_decoded_error_q99": 127.0,
  "canonical_decoded_exact_value_transition_accuracy": 0.05838666078225851,
  "canonical_decoded_maximum_absolute_error": 127,
  "canonical_decoded_median_performance_mae": 27.831368899917287,
  "canonical_decoded_on_region_mae": 19.13127643141394,
  "canonical_decoded_overall_mae": 28.378264031726353,
  "canonical_decoded_pedal1_mae": 28.386013355498577,
  "canonical_decoded_pedal2_mae": 28.4854928009918,
  "canonical_decoded_pedal3_mae": 28.268226451776506,
  "canonical_decoded_pedal4_mae": 28.37332351863853,
  "canonical_decoded_performance_macro_mae": 26.531018428623486,
  "canonical_decoded_steady_position_accuracy": 0.5057534384858045,
  "canonical_decoded_subthreshold_mae": 42.190472737293234,
  "canonical_decoded_tolerance_10": 0.4970001197486687,
  "canonical_decoded_tolerance_20": 0.5838645712997661,
  "canonical_decoded_tolerance_5": 0.4765900157786481,
  "canonical_decoded_transition_direction_accuracy": 0.151369504054725,
  "canonical_decoded_transition_f1": 0.20604829246808523,
  "canonical_decoded_transition_precision": 0.1929995724102212,
  "canonical_decoded_transition_recall": 0.22098940806531692,
  "canonical_decoded_zero_mae": 44.876062726345765,
  "canonical_oracle_median_performance_mae": 3.4748825352394284,
  "canonical_oracle_on_region_mae": 2.5971043014957886,
  "canonical_oracle_overall_mae": 3.635270209348849,
  "canonical_oracle_pedal1_mae": 3.654623707418782,
  "canonical_oracle_pedal2_mae": 3.6714624834465077,
  "canonical_oracle_pedal3_mae": 3.616568284917303,
  "canonical_oracle_pedal4_mae": 3.598426361612803,
  "canonical_oracle_performance_macro_mae": 4.171420050076859,
  "canonical_oracle_subthreshold_mae": 16.503658474680744,
  "checkpoint": "/workspace/project/analysis/stage2_encoder_only_ordinal_v0/lambda_1p0/best.pt",
  "decoder": "smallest 128-class posterior cumulative value reaching 0.5",
  "five_class_exact_note_accuracy": 0.42725268377898623,
  "five_class_macro_f1": 0.37157399335671326,
  "five_class_micro_f1": 0.5160304725141586,
  "five_class_token_accuracy": 0.5160304725141586,
  "five_class_weighted_f1": 0.5163253010637854,
  "full_f1": 0.735975288913141,
  "full_precision": 0.7120439084724799,
  "full_predicted_count": 530376,
  "full_recall": 0.7615712545676004,
  "full_support": 495884,
  "high_f1": 0.14126633791619644,
  "high_precision": 0.12636133074742653,
  "high_predicted_count": 100545,
  "high_recall": 0.16015782573618395,
  "high_support": 79328,
  "high_to_mid_count": 12777,
  "high_to_mid_rate": 0.16106544977813633,
  "is_new_model": false,
  "is_primary_reference": false,
  "label": "Ordinal lambda 1.0 posterior median",
  "low_f1": 0.2146100098497907,
  "low_precision": 0.2285889922544158,
  "low_predicted_count": 122005,
  "low_recall": 0.20224222075577053,
  "low_support": 137899,
  "low_to_mid_count": 27901,
  "low_to_mid_rate": 0.2023292409662144,
  "low_to_zero_count": 52587,
  "low_to_zero_rate": 0.3813443172176738,
  "mid_f1": 0.25458856176793493,
  "mid_high_full_aggregate_recall": 0.8817078145171803,
  "mid_precision": 0.23810813705171252,
  "mid_predicted_count": 158743,
  "mid_recall": 0.27351998321164184,
  "mid_support": 138191,
  "mid_to_high_count": 21554,
  "mid_to_high_rate": 0.1559725307726263,
  "model": "ordinal_lambda1_posterior_median",
  "model_minus_oracle_on_region_mae": 16.53417212991815,
  "model_minus_oracle_overall_mae": 24.742993822377503,
  "nonendpoint_endpoint_collapse_ratio": 0.5085054780568232,
  "off_f1": 0.6810844438197348,
  "off_on_balanced_accuracy": 0.7506483942337671,
  "off_on_state_accuracy": 0.7842401946972472,
  "off_on_state_macro_f1": 0.7590285568503591,
  "off_precision": 0.756132097281302,
  "off_recall": 0.6195889739503538,
  "off_to_on_transition_f1": 0.05828010636256476,
  "off_to_on_transition_precision": 0.053697716710446555,
  "off_to_on_transition_predicted_count": 19796,
  "off_to_on_transition_recall": 0.06371755679434155,
  "off_to_on_transition_true_count": 16683,
  "off_to_on_transition_true_positive": 1063,
  "on_f1": 0.8369726698809834,
  "on_precision": 0.7965577764720185,
  "on_recall": 0.8817078145171803,
  "on_region_3way_accuracy": 0.6001572743596537,
  "on_region_macro_f1": 0.42674088089731416,
  "on_to_off_transition_f1": 0.0573341014456972,
  "on_to_off_transition_precision": 0.05288193917311877,
  "on_to_off_transition_predicted_count": 19761,
  "on_to_off_transition_recall": 0.06260484064222382,
  "on_to_off_transition_true_count": 16692,
  "on_to_off_transition_true_positive": 1045,
  "predicted_full_ratio": 0.4669986757206052,
  "predicted_high_ratio": 0.0885303668535685,
  "predicted_low_ratio": 0.10742600236679721,
  "predicted_mid_ratio": 0.1397739919979713,
  "predicted_off_ratio": 0.30469696542785496,
  "predicted_on_ratio": 0.695303034572145,
  "predicted_zero_ratio": 0.19727096306105774,
  "short_repedal_f1": 0.014594623598218599,
  "short_repedal_precision": 0.011197101926560184,
  "short_repedal_predicted_count": 12146,
  "short_repedal_recall": 0.02095208750577723,
  "short_repedal_true_count": 6491,
  "short_repedal_true_positive": 136,
  "steady_position_state_accuracy": 0.7925155996828351,
  "target_full_ratio": 0.4366283001324279,
  "target_high_ratio": 0.06984869403510749,
  "target_low_ratio": 0.12142074751345412,
  "target_mid_ratio": 0.12167785494914203,
  "target_zero_ratio": 0.25042440336986843,
  "transition_direction_accuracy": 0.06316104868913858,
  "zero_f1": 0.5114297683365031,
  "zero_low_aggregate_recall": 0.6195889739503538,
  "zero_precision": 0.5803305615439893,
  "zero_predicted_count": 224043,
  "zero_recall": 0.45715340529517245,
  "zero_support": 284410,
  "zero_to_low_count": 51163,
  "zero_to_low_rate": 0.1798917056362294,
  "zero_vs_nonzero_accuracy": 0.7812693711081683
}
```

Coarse-to-fine v0:

```json
{
  "all_five_classes_predicted": true,
  "architecture": "official PT encoder + four region heads + four depth heads",
  "best_epoch": 3,
  "best_validation_loss": 0.9594596367820074,
  "binary_transition_f1": 0.11421661909292738,
  "binary_transition_precision": 0.08533641385642986,
  "binary_transition_recall": 0.17264419475655432,
  "canonical_decoded_delta_mae": 8.549148014205194,
  "canonical_decoded_error_median": 0.0,
  "canonical_decoded_error_q01": 0.0,
  "canonical_decoded_error_q05": 0.0,
  "canonical_decoded_error_q25": 0.0,
  "canonical_decoded_error_q75": 47.0,
  "canonical_decoded_error_q95": 127.0,
  "canonical_decoded_error_q99": 127.0,
  "canonical_decoded_exact_value_transition_accuracy": 0.07588128206542727,
  "canonical_decoded_maximum_absolute_error": 127,
  "canonical_decoded_median_performance_mae": 27.966684804115552,
  "canonical_decoded_on_region_mae": 21.018438386157612,
  "canonical_decoded_overall_mae": 28.329023555267533,
  "canonical_decoded_pedal1_mae": 28.249774590741314,
  "canonical_decoded_pedal2_mae": 28.299322363416078,
  "canonical_decoded_pedal3_mae": 28.297357076441916,
  "canonical_decoded_pedal4_mae": 28.469640190470823,
  "canonical_decoded_performance_macro_mae": 27.00093682384151,
  "canonical_decoded_steady_position_accuracy": 0.5639984858044164,
  "canonical_decoded_subthreshold_mae": 43.320393911485944,
  "canonical_decoded_tolerance_10": 0.5493478980586628,
  "canonical_decoded_tolerance_20": 0.5946578005691584,
  "canonical_decoded_tolerance_5": 0.5297725127497112,
  "canonical_decoded_transition_direction_accuracy": 0.14104650521321785,
  "canonical_decoded_transition_f1": 0.21998445105213318,
  "canonical_decoded_transition_precision": 0.23619891601060253,
  "canonical_decoded_transition_recall": 0.20585314723892537,
  "canonical_decoded_zero_mae": 39.39789740163848,
  "canonical_oracle_median_performance_mae": 3.4748825352394284,
  "canonical_oracle_on_region_mae": 2.5971043014957886,
  "canonical_oracle_overall_mae": 3.635270209348849,
  "canonical_oracle_pedal1_mae": 3.654623707418782,
  "canonical_oracle_pedal2_mae": 3.6714624834465077,
  "canonical_oracle_pedal3_mae": 3.616568284917303,
  "canonical_oracle_pedal4_mae": 3.598426361612803,
  "canonical_oracle_performance_macro_mae": 4.171420050076859,
  "canonical_oracle_subthreshold_mae": 16.503658474680744,
  "checkpoint": "/workspace/project/analysis/stage2_encoder_only_coarse_to_fine_v0/best.pt",
  "decoder": "region argmax plus conditional sigmoid depth after raw-output averaging",
  "five_class_exact_note_accuracy": 0.47322560649178663,
  "five_class_macro_f1": 0.36089797606612317,
  "five_class_micro_f1": 0.5598320701022794,
  "five_class_token_accuracy": 0.5598320701022794,
  "five_class_weighted_f1": 0.5286817292910947,
  "full_f1": 0.7503607342511938,
  "full_precision": 0.6964993514758889,
  "full_predicted_count": 579007,
  "full_recall": 0.8132506795944213,
  "full_support": 495884,
  "high_f1": 0.0033284896357291013,
  "high_precision": 0.11269974768713205,
  "high_predicted_count": 1189,
  "high_recall": 0.0016891891891891893,
  "high_support": 79328,
  "high_to_mid_count": 18878,
  "high_to_mid_rate": 0.2379739814441307,
  "is_new_model": false,
  "is_primary_reference": false,
  "label": "Coarse-to-fine v0",
  "low_f1": 0.18955387939203275,
  "low_precision": 0.2734094170710039,
  "low_predicted_count": 73165,
  "low_recall": 0.1450626908099406,
  "low_support": 137899,
  "low_to_mid_count": 29366,
  "low_to_mid_rate": 0.21295295832457087,
  "low_to_zero_count": 64676,
  "low_to_zero_rate": 0.46900992755567483,
  "mid_f1": 0.29416304931738607,
  "mid_high_full_aggregate_recall": 0.8659831820163358,
  "mid_precision": 0.2565276154885817,
  "mid_predicted_count": 185711,
  "mid_recall": 0.34474025081228155,
  "mid_support": 138191,
  "mid_to_high_count": 474,
  "mid_to_high_rate": 0.0034300352410793758,
  "model": "coarse_to_fine_v0",
  "model_minus_oracle_on_region_mae": 18.421334084661822,
  "model_minus_oracle_overall_mae": 24.693753345918683,
  "nonendpoint_endpoint_collapse_ratio": 0.619898260639585,
  "off_f1": 0.6923170149751172,
  "off_on_balanced_accuracy": 0.7576318425775164,
  "off_on_state_accuracy": 0.78540334169226,
  "off_on_state_macro_f1": 0.7637822645094134,
  "off_precision": 0.7414637444058355,
  "off_recall": 0.649280503138697,
  "off_to_on_transition_f1": 0.09536455339978993,
  "off_to_on_transition_precision": 0.07123401231643771,
  "off_to_on_transition_predicted_count": 33776,
  "off_to_on_transition_recall": 0.14421866570760655,
  "off_to_on_transition_true_count": 16683,
  "off_to_on_transition_true_positive": 2406,
  "on_f1": 0.8352475140437096,
  "on_precision": 0.8066188192561238,
  "on_recall": 0.8659831820163358,
  "on_region_3way_accuracy": 0.632254139665799,
  "on_region_macro_f1": 0.3946960178798489,
  "on_to_off_transition_f1": 0.07581735630588655,
  "on_to_off_transition_precision": 0.0566602459623648,
  "on_to_off_transition_predicted_count": 33745,
  "on_to_off_transition_recall": 0.11454589024682482,
  "on_to_off_transition_true_count": 16692,
  "on_to_off_transition_true_positive": 1912,
  "predicted_full_ratio": 0.5098185103265617,
  "predicted_high_ratio": 0.001046920346003212,
  "predicted_low_ratio": 0.06442214223324223,
  "predicted_mid_ratio": 0.16351944859260095,
  "predicted_off_ratio": 0.3256151207348342,
  "predicted_on_ratio": 0.6743848792651658,
  "predicted_zero_ratio": 0.26119297850159195,
  "short_repedal_f1": 0.028976994615761135,
  "short_repedal_precision": 0.018382048521983937,
  "short_repedal_predicted_count": 24154,
  "short_repedal_recall": 0.06840240332768449,
  "short_repedal_true_count": 6491,
  "short_repedal_true_positive": 444,
  "steady_position_state_accuracy": 0.7929810045851001,
  "target_full_ratio": 0.4366283001324279,
  "target_high_ratio": 0.06984869403510749,
  "target_low_ratio": 0.12142074751345412,
  "target_mid_ratio": 0.12167785494914203,
  "target_zero_ratio": 0.25042440336986843,
  "transition_direction_accuracy": 0.12937827715355804,
  "zero_f1": 0.5670837277342742,
  "zero_low_aggregate_recall": 0.649280503138697,
  "zero_precision": 0.555393743257821,
  "zero_predicted_count": 296640,
  "zero_recall": 0.579276396751169,
  "zero_support": 284410,
  "zero_to_low_count": 24765,
  "zero_to_low_rate": 0.08707499736296193,
  "zero_vs_nonzero_accuracy": 0.778512510213857
}
```

## 22. Success criteria 판정

```json
{
  "all_safeguards_passed": true,
  "criterion_a": false,
  "criterion_a_checks": {
    "binary_transition_f1_decrease_at_most_0p02": true,
    "macro_f1_improvement_at_least_0p03": false,
    "on_region_mae_worsening_at_most_1p0": false
  },
  "criterion_b": false,
  "criterion_b_checks": {
    "binary_transition_f1_improvement_at_least_0p03": true,
    "macro_f1_decrease_at_most_0p01": false,
    "on_region_mae_improvement_at_least_1p0": false
  },
  "deltas_new_minus_reference": {
    "binary_transition_f1": 0.0546237211711605,
    "five_class_macro_f1": -0.029741797204937914,
    "on_region_mae": 2.7882599316235037
  },
  "passed": false,
  "primary_reference": "ce_128_posterior_median",
  "safeguards": {
    "all_five_classes_predicted": true,
    "binary_transition_f1_decrease_at_most_0p02": true,
    "duplicate_evaluation_absent": true,
    "low_predicted_ratio_nonzero": true,
    "no_class_recall_exactly_zero": true,
    "not_collapsed_to_one_binary_state": true,
    "output_finite": true,
    "predicted_off_ratio_in_0p10_to_0p80": true,
    "predicted_on_ratio_in_0p20_to_0p90": true,
    "validation_notes_aggregated_once": true
  }
}
```

## 23. 과학적 해석

- All-class recall diagnostic: no class has exactly zero recall.
- ZERO/LOW separation: LOW recall=0.084678; detailed directional confusion is reported above.
- Renderer state versus depth localization: OFF/ON accuracy=0.790821, ON-region 3-way accuracy=0.638821.
- Temporal transition diagnostic: binary transition F1=0.127102 versus reference 0.072479.
- Quantization/model decomposition: oracle MAE=3.635270, model MAE=28.764687, gap=25.129417.
- Predeclared decision: Criterion A=False, Criterion B=False, safeguards=True, final pass=False.

## 24. 한계

- Model selection과 결과 해석은 ASAP validation에 한정된다.
- LOW 내부 raw-value error는 OFF-state semantics와 quantization error를 함께 반영한다.
- Four pedal slots are conditionally predicted from encoder states without explicit previous-pedal conditioning or smoothing.
- Short repedal은 variable-duration IOI를 가진 sample-count diagnostic이다.

## 25. 권장 다음 단계

사전 기준을 통과하지 못했으므로 자동으로 test/MAESTRO/추가 architecture 실험을 진행하지 않고 측정된 failure mode를 먼저 검토한다.

## 26. ASAP test split 미사용 확인

- Evaluation split: `validation`
- Evaluation test rows used: `0`
- Training/evaluation pipeline test rows: `0`
- Selected unit tests excluded `test_stage2_dataset.py`, whose real-data smoke includes ASAP test MIDI.
- This run generated no ASAP test metric, prediction, representative, checkpoint selection, MIDI, or report input.
