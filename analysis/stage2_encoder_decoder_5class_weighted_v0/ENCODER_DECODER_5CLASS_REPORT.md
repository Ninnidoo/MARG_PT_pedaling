# Stage 2 Encoder-Decoder Weighted 5-class Report

## Scope

Corrected weighted encoder-only baseline과 데이터, five-class target, train-only inverse-sqrt
weighted CE, optimizer, window/stride, early stopping을 유지하고 Stage 2 architecture만
PT-native encoder-decoder로 변경했다. ASAP test, MAESTRO, 23-score inference는 사용하지 않았다.

## Architecture

official pretrained unfrozen PT encoder (10x768) + fresh PT-native 2-layer causal decoder (hidden 768, FFN 3072, head dim 128) + Pedal1-4 slot embeddings + shared Linear(768,5)

- Encoder: pretrained official PT encoder, 10 layers, hidden 768, unfrozen
- Decoder: fresh seed `42`, 2 causal layers, hidden 768,
  FFN 3072, head dimension 128, native rotary position/norm/dropout
- Sequence: `P1_1,P2_1,P3_1,P4_1,...`; teacher input is shifted right with BOS
- Decoder vocabulary: five classes + BOS + PAD; shared `Linear(768,5)` output

## Controlled configuration

```json
{
  "amp_enabled": true,
  "amp_init_scale": 1024.0,
  "architecture": "official pretrained unfrozen PT encoder (10x768) + fresh PT-native 2-layer causal decoder (hidden 768, FFN 3072, head dim 128) + Pedal1-4 slot embeddings + shared Linear(768,5)",
  "asap_root": "/workspace/public/ASAP/asap-dataset-v1.1",
  "checkpoint_path": "/workspace/project/checkpoints/pianist_transformer",
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
  "class_names": [
    "ZERO",
    "LOW",
    "MID",
    "HIGH",
    "FULL"
  ],
  "container_hostname": "d3affcb5b8ad",
  "container_visible_cuda_index": 0,
  "cuda_version": "12.6",
  "decoder_init_seed": 42,
  "decoder_initial_sha256": "4fd858b3213aa4a9f11ad3b41e8ace2665f0595ddef3b8c2c03279dbb3892683",
  "decoder_lr": 0.0001,
  "early_stopping_min_delta": 0.0001,
  "early_stopping_patience": 4,
  "effective_batch_size": 16,
  "encoder_initial_sha256": "4bfcd1c68006b148799675b126495e9af750215ae5ed77d63621f6139da2ebe4",
  "encoder_lr": 1e-05,
  "expected_gpu_uuid": "GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b",
  "expected_train_performances": 892,
  "expected_train_pieces": 180,
  "expected_validation_performances": 71,
  "expected_validation_pieces": 19,
  "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
  "gpu_uuid": "GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b",
  "gradient_accumulation_steps": 4,
  "loss_configuration": {
    "class_weighting": "global_train_pedal1_4_inverse_sqrt_probability",
    "class_weights": [
      0.9258126368700947,
      1.204207482868018,
      1.1252359220805253,
      1.7123712206707022,
      0.7669776275315598
    ],
    "ignore_index": -100,
    "label_smoothing": 0.0,
    "name": "class_weighted_cross_entropy_inverse_sqrt_train_global",
    "normalization": "sum_c p_c w_c = 1"
  },
  "max_epochs": 20,
  "max_grad_norm": 1.0,
  "micro_batch_size": 4,
  "model_selection": "teacher_forced_validation_weighted_ce",
  "num_workers": 0,
  "output_dir": "/workspace/project/analysis/stage2_encoder_decoder_5class_weighted_v0",
  "output_head_initial_sha256": "d1a5593367d8b3f1f57d0713fc561bba712ab6a0dd4c4196a52b10a225daf0d3",
  "pin_memory": true,
  "pipeline_splits": [
    "train",
    "validation"
  ],
  "progress_interval": 100,
  "pytorch_version": "2.8.0+cu126",
  "representatives": [
    0,
    32,
    80,
    111,
    127
  ],
  "run_id": "20260807T070913Z-97577623ff04",
  "seed": 42,
  "slot_embedding_initial_sha256": "b230966f338491d0a6d3fbfc3c02abb4d86c5c3d10fba97f78e4f7e8f9edd7ae",
  "split_csv": "/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv",
  "stride_notes": 256,
  "test_split_passed_to_pipeline": false,
  "tmux_session_name": "stage2-encoder-decoder-5class-weighted-v0",
  "train_performance_count": 892,
  "train_piece_count": 180,
  "train_preload_seconds": 62.013363029807806,
  "train_window_count": 11230,
  "validation_performance_count": 71,
  "validation_piece_count": 19,
  "validation_preload_seconds": 5.457137547433376,
  "validation_window_count": 1078,
  "weight_decay": 0.01,
  "window_notes": 512
}
```

## Tests

```json
{
  "asap_test_split_accessed": false,
  "errors": [],
  "excluded_real_dataset_test": "tests.test_stage2_dataset",
  "failures": [],
  "modules": [
    "tests.test_stage2_encoder_decoder_five_class",
    "tests.test_stage2_weighted_five_class",
    "tests.test_stage2_five_class",
    "tests.test_stage2_five_class_runtime"
  ],
  "passed": true,
  "skipped": [],
  "total_tests": 33
}
```

## GPU memory smoke and effective batch

```json
{
  "attempts": [
    {
      "finite": false,
      "micro_batch_size": 16,
      "oom": true
    },
    {
      "finite": false,
      "micro_batch_size": 8,
      "oom": true
    },
    {
      "finite": true,
      "micro_batch_size": 4,
      "oom": false,
      "peak_gpu_memory_bytes": 7409522176
    }
  ],
  "effective_batch_size": 16,
  "gradient_accumulation_steps": 4,
  "passed": true,
  "selected_micro_batch_size": 4,
  "test_split_accessed": false
}
```

## Pedal-rich tiny overfit

```json
{
  "final_loss": 0.0497452348238197,
  "final_teacher_forced_accuracy": 1.0,
  "gradient_groups_finite_nonzero": {
    "decoder": true,
    "encoder": true,
    "output": true,
    "slot_embedding": true
  },
  "initial_loss": 2.2355160205154503,
  "note_count": 16,
  "passed": true,
  "performance_path": "Schubert/Piano_Sonatas/664-3/Lin07.mid",
  "predicted_class_counts": [
    1,
    16,
    16,
    12,
    19
  ],
  "source_split": "train",
  "steps": 28,
  "target_class_counts": [
    1,
    16,
    16,
    12,
    19
  ],
  "test_split_accessed": false,
  "window_start": 326
}
```

## Training

Best checkpoint criterion is minimum teacher-forced validation weighted CE.

```json
{
  "best_checkpoint_rule": "minimum teacher-forced validation weighted CE",
  "best_epoch": 4,
  "best_validation_loss": 0.27912840524797056,
  "completed_epoch": 8,
  "early_stopped": true,
  "effective_batch_size": 16,
  "global_optimizer_step": 5616,
  "gradient_accumulation_steps": 4,
  "micro_batch_size": 4,
  "repair": "validation/report rerun only; no training or checkpoint modification",
  "stop_reason": "early_stopping"
}
```

## Primary validation comparison

Greedy free-running autoregressive decoding is the primary encoder-decoder result.

```json
{
  "binary_transition_f1": {
    "corrected_weighted_encoder_only": "0.12942971923603475",
    "encoder_decoder_greedy_free_running": 0.003870013089750157,
    "encoder_decoder_teacher_forced": 0.17464178705983563
  },
  "canonical_decoded_on_region_mae": {
    "corrected_weighted_encoder_only": "22.675613923686893",
    "encoder_decoder_greedy_free_running": 31.710603964379178,
    "encoder_decoder_teacher_forced": 4.678969670719074
  },
  "canonical_decoded_overall_mae": {
    "corrected_weighted_encoder_only": "28.605620086782565",
    "encoder_decoder_greedy_free_running": 40.50015761038009,
    "encoder_decoder_teacher_forced": 5.868646276520808
  },
  "canonical_decoded_subthreshold_mae": {
    "corrected_weighted_encoder_only": "45.47170755407944",
    "encoder_decoder_greedy_free_running": 55.102306760745186,
    "encoder_decoder_teacher_forced": 19.207905786118825
  },
  "five_class_macro_f1": {
    "corrected_weighted_encoder_only": "0.3634246297618995",
    "encoder_decoder_greedy_free_running": 0.26142154119576005,
    "encoder_decoder_teacher_forced": 0.9045459580124756
  },
  "five_class_token_accuracy": {
    "corrected_weighted_encoder_only": "0.5708454256008566",
    "encoder_decoder_greedy_free_running": 0.49343407483587387,
    "encoder_decoder_teacher_forced": 0.9365930799357584
  },
  "nonendpoint_endpoint_collapse_ratio": {
    "corrected_weighted_encoder_only": "0.7118238243420423",
    "encoder_decoder_greedy_free_running": 0.8936210321368079,
    "encoder_decoder_teacher_forced": 0.0661530929778458
  },
  "off_on_state_accuracy": {
    "corrected_weighted_encoder_only": "0.789293412414415",
    "encoder_decoder_greedy_free_running": 0.6929855456312868,
    "encoder_decoder_teacher_forced": 0.9729015806824265
  },
  "off_on_state_macro_f1": {
    "corrected_weighted_encoder_only": "0.7739101052189711",
    "encoder_decoder_greedy_free_running": 0.6706361914345249,
    "encoder_decoder_teacher_forced": 0.9709947553459788
  },
  "short_repedal_f1": {
    "corrected_weighted_encoder_only": "0.044601421002915934",
    "encoder_decoder_greedy_free_running": 0.0,
    "encoder_decoder_teacher_forced": 0.013605442176870748
  }
}
```

## Teacher-forced versus free-running gap

```json
{
  "free_minus_teacher_binary_transition_f1": -0.17077177397008547,
  "free_minus_teacher_canonical_decoded_on_region_mae": 27.031634293660105,
  "free_minus_teacher_canonical_decoded_overall_mae": 34.63151133385929,
  "free_minus_teacher_canonical_decoded_subthreshold_mae": 35.894400974626365,
  "free_minus_teacher_five_class_macro_f1": -0.6431244168167156,
  "free_minus_teacher_five_class_token_accuracy": -0.4431590050998845,
  "free_minus_teacher_nonendpoint_endpoint_collapse_ratio": 0.8274679391589621,
  "free_minus_teacher_off_on_state_accuracy": -0.2799160350511397,
  "free_minus_teacher_off_on_state_macro_f1": -0.3003585639114539,
  "free_minus_teacher_short_repedal_f1": -0.013605442176870748
}
```

## Free-running per-class precision/recall/F1 and distribution

```json
{
  "all_five_classes_predicted": false,
  "confusion_matrix": "[[149579  13168   2065      0 119598]\n [ 71772  11075   4041      0  51011]\n [ 48536  11877   5456      0  72322]\n [ 19705   3536   1824      0  54263]\n [ 77422  10889  13284      0 394289]]",
  "exact_note_accuracy": 0.45917979205995885,
  "f1": "[0.45923699 0.11754155 0.06618909 0.         0.66414007]",
  "full_recall": 0.7951234562921974,
  "high_recall": 0.0,
  "high_to_mid_count": 1824,
  "high_to_mid_rate": 0.022993142396127472,
  "low_recall": 0.08031240255549352,
  "low_to_mid_count": 4041,
  "low_to_mid_rate": 0.029304055866975105,
  "low_to_zero_count": 71772,
  "low_to_zero_rate": 0.5204678786648199,
  "macro_f1": 0.26142154119576005,
  "macro_precision": 0.2802901141154658,
  "macro_recall": 0.2881689608634999,
  "micro_f1": 0.49343407483587387,
  "micro_precision": 0.49343407483587387,
  "micro_recall": 0.49343407483587387,
  "mid_high_full_aggregate_recall": 0.7589511117839426,
  "mid_recall": 0.03948158707875332,
  "mid_to_high_count": 0,
  "mid_to_high_rate": 0.0,
  "nonendpoint_endpoint_collapse_ratio": 0.8936210321368079,
  "precision": "[0.40755666 0.21911168 0.20457443 0.         0.5702078 ]",
  "predicted_count": "[367014  50545  26670      0 691483]",
  "predicted_distribution": "[0.32315763 0.04450512 0.02348307 0.         0.60885418]",
  "recall": "[0.52592736 0.0803124  0.03948159 0.         0.79512346]",
  "support": "[284410 137899 138191  79328 495884]",
  "target_count": 1135712,
  "target_distribution": "[0.2504244  0.12142075 0.12167785 0.06984869 0.4366283 ]",
  "token_accuracy": 0.49343407483587387,
  "valid_note_count": 283928,
  "weighted_f1": 0.42731222906057953,
  "weighted_precision": 0.40252787763921405,
  "weighted_recall": 0.49343407483587387,
  "zero_low_aggregate_recall": 0.5815504760732071,
  "zero_recall": 0.5259273583910552,
  "zero_to_low_count": 13168,
  "zero_to_low_rate": 0.04629935656270877
}
```

## Free-running OFF/ON, transition timing, and short-repedal metrics

```json
{
  "combined_binary_transition": {
    "f1": 0.003870013089750157,
    "precision": 0.038483305036785515,
    "predicted_count": 1767,
    "recall": 0.002037453183520599,
    "true_count": 33375,
    "true_positive": 68
  },
  "off_to_on_transition": {
    "f1": 0.00216179315052907,
    "precision": 0.021229050279329607,
    "predicted_count": 895,
    "recall": 0.0011388838937840915,
    "true_count": 16683,
    "true_positive": 19
  },
  "on_region_depth_class_accuracy": 0.5603354625646374,
  "on_region_macro_f1": 0.2814385532237484,
  "on_region_per_class_f1": {
    "FULL": 0.7755808166741741,
    "HIGH": 0.0,
    "MID": 0.06873484299707096
  },
  "on_to_off_transition": {
    "f1": 0.0012525620587565474,
    "precision": 0.01261467889908257,
    "predicted_count": 872,
    "recall": 0.0006589983225497244,
    "true_count": 16692,
    "true_positive": 11
  },
  "sequence_order": "Within each performance, note-major Pedal1->Pedal2->Pedal3->Pedal4; no cross-performance boundary.",
  "short_repedal_like": {
    "definition": ">=64 -> <64 -> >=64 with OFF run <=4 flattened samples; exact recall requires both transition boundaries at the same samples",
    "f1": 0.0,
    "precision": 0.0,
    "predicted_count": 770,
    "recall": 0.0,
    "sample_based_not_milliseconds": true,
    "true_count": 6491,
    "true_positive": 0
  },
  "state": {
    "accuracy": 0.6929855456312868,
    "balanced_accuracy": 0.6702507939285749,
    "confusion_matrix": "[[245594 176715]\n [171965 541438]]",
    "macro_f1": 0.6706361914345249,
    "off_f1": 0.5848395224011392,
    "off_precision": 0.5881659837292454,
    "off_recall": 0.5815504760732071,
    "off_support": 422309,
    "on_f1": 0.7564328604679105,
    "on_precision": 0.753931265343179,
    "on_recall": 0.7589511117839426,
    "on_support": 713403,
    "predicted_off_count": 417559,
    "predicted_on_count": 718153
  },
  "steady_position_state_accuracy": 0.6988267804685984,
  "steady_position_state_count": 1102266,
  "transition_direction_accuracy": 0.0008988764044943821,
  "transition_timing_sample_offset": {
    "matching": "earliest-feasible one-to-one, within performance and direction",
    "plus_minus_0_samples": {
      "combined": {
        "f1": 0.0017073587160662455,
        "matched_count": 30,
        "maximum_absolute_offset": 0,
        "mean_absolute_offset": 0.0,
        "mean_signed_offset": 0.0,
        "median_absolute_offset": 0.0,
        "precision": 0.01697792869269949,
        "predicted_count": 1767,
        "recall": 0.0008988764044943821,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.00216179315052907,
        "matched_count": 19,
        "precision": 0.021229050279329607,
        "predicted_count": 895,
        "recall": 0.0011388838937840915,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.0012525620587565474,
        "matched_count": 11,
        "precision": 0.01261467889908257,
        "predicted_count": 872,
        "recall": 0.0006589983225497244,
        "true_count": 16692
      }
    },
    "plus_minus_1_samples": {
      "combined": {
        "f1": 0.005235900062603153,
        "matched_count": 92,
        "maximum_absolute_offset": 1,
        "mean_absolute_offset": 0.6739130434782609,
        "mean_signed_offset": -0.2391304347826087,
        "median_absolute_offset": 1.0,
        "precision": 0.0520656479909451,
        "predicted_count": 1767,
        "recall": 0.0027565543071161047,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.005688929343497554,
        "matched_count": 50,
        "precision": 0.055865921787709494,
        "predicted_count": 895,
        "recall": 0.0029970628783791884,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.004782509678888637,
        "matched_count": 42,
        "precision": 0.0481651376146789,
        "predicted_count": 872,
        "recall": 0.0025161754133716753,
        "true_count": 16692
      }
    },
    "plus_minus_2_samples": {
      "combined": {
        "f1": 0.006715610949860566,
        "matched_count": 118,
        "maximum_absolute_offset": 2,
        "mean_absolute_offset": 1.0,
        "mean_signed_offset": -0.559322033898305,
        "median_absolute_offset": 1.0,
        "precision": 0.06677985285795134,
        "predicted_count": 1767,
        "recall": 0.0035355805243445695,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.006826715212197064,
        "matched_count": 60,
        "precision": 0.0670391061452514,
        "predicted_count": 895,
        "recall": 0.0035964754540550262,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.006604418127989068,
        "matched_count": 58,
        "precision": 0.06651376146788991,
        "predicted_count": 872,
        "recall": 0.003474718427989456,
        "true_count": 16692
      }
    },
    "plus_minus_4_samples": {
      "combined": {
        "f1": 0.007853850093904729,
        "matched_count": 138,
        "maximum_absolute_offset": 4,
        "mean_absolute_offset": 2.391304347826087,
        "mean_signed_offset": -1.9710144927536233,
        "median_absolute_offset": 2.0,
        "precision": 0.07809847198641766,
        "predicted_count": 1767,
        "recall": 0.004134831460674157,
        "true_count": 33375
      },
      "off_to_on": {
        "f1": 0.00830583684150643,
        "matched_count": 73,
        "precision": 0.08156424581005586,
        "predicted_count": 895,
        "recall": 0.004375711802433615,
        "true_count": 16683
      },
      "on_to_off": {
        "f1": 0.007401503074470508,
        "matched_count": 65,
        "precision": 0.07454128440366972,
        "predicted_count": 872,
        "recall": 0.003894080996884735,
        "true_count": 16692
      }
    }
  },
  "zero_vs_nonzero": {
    "accuracy": 0.6898280549998591,
    "balanced_accuracy": 0.6352563555900386,
    "confusion_matrix": "[[633867 217435]\n [134831 149579]]",
    "macro_f1": 0.6208944232487141,
    "off_f1": 0.7825518518518518,
    "off_precision": 0.8245982167249036,
    "off_recall": 0.7445853527890219,
    "off_support": 851302,
    "on_f1": 0.4592369946455765,
    "on_precision": 0.4075566599639251,
    "on_recall": 0.5259273583910552,
    "on_support": 284410,
    "predicted_off_count": 768698,
    "predicted_on_count": 367014
  }
}
```

## Free-running decoded MAE

```json
{
  "decoded_transition_direction_accuracy": 0.002261819385447123,
  "decoded_transition_f1": 0.008625285990996765,
  "decoded_transition_precision": 0.20266412940057088,
  "decoded_transition_recall": 0.004406410327136316,
  "delta_mae": 3.331313328772033,
  "error_quantiles": {
    "median": 9.0,
    "q01": 0.0,
    "q05": 0.0,
    "q25": 0.0,
    "q75": 76.0,
    "q95": 127.0,
    "q99": 127.0
  },
  "exact_value_transition_accuracy": 0.08658354940144536,
  "maximum_absolute_error": 127,
  "median_performance_mae": 40.39387119113574,
  "on_region_mae": 31.710603964379178,
  "overall_micro_mae": 40.50015761038009,
  "per_slot_mae": {
    "Pedal1": 40.426326392606576,
    "Pedal2": 40.65014017638274,
    "Pedal3": 40.52795779211631,
    "Pedal4": 40.39620608041475
  },
  "performance_count": 71,
  "performance_macro_mae": 37.03757286646199,
  "steady_position_decoded_accuracy": 0.5365794321766562,
  "subthreshold_region_mae": 55.102306760745186,
  "target_count": 1135712,
  "tolerance_accuracy": {
    "plus_minus_10": 0.5026644078780536,
    "plus_minus_20": 0.5252969062579246,
    "plus_minus_5": 0.49504539883350707
  },
  "zero_target_mae": 55.46753630322422
}
```

## Validation-only provenance

- Split: `validation`
- Test rows used: `0`
- Performances/pieces/notes/windows: `71` /
  `19` / `283928` / `1078`
- Class weights source: train Pedal1–4 only; `Σ p_c w_c = 1`
