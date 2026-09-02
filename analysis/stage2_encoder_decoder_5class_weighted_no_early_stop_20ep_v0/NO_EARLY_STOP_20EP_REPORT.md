# Encoder-Decoder Weighted 5-class: No Early Stop 20-Epoch Trajectory

## Scope

기존 completed encoder-decoder weighted 5-class run과 동일한 초기화 및 training
configuration으로 처음부터 다시 학습했다. Early stopping 종료 동작만 비활성화했으며,
best checkpoint selection은 validation teacher-forced weighted CE 기준으로 유지했다.
ASAP test와 free-running full validation은 사용하지 않았다.

## Summary

- Completed epochs: `20`
- Best validation epoch/loss: `4` / `0.27912840524797056`
- Epoch 20 validation loss: `0.5073045373148711`
- Epoch 20 train loss: `0.13264172313896938`
- Best checkpoint matches global minimum: `True`
- Validation rose and later declined after best: `False`
- Train-down/validation-up overfitting region present: `True`
- Baseline stopped at epoch: `8`
- Baseline early stopping was too early for the observed global best: `False`

## Post-best rises

```json
[
  {
    "delta": 0.0007516794942339367,
    "from_epoch": 4,
    "to_epoch": 5
  },
  {
    "delta": 0.0005200263066091892,
    "from_epoch": 5,
    "to_epoch": 6
  },
  {
    "delta": 0.00035089442815455785,
    "from_epoch": 6,
    "to_epoch": 7
  },
  {
    "delta": 0.0022280439886248615,
    "from_epoch": 7,
    "to_epoch": 8
  },
  {
    "delta": 0.002210697589299704,
    "from_epoch": 8,
    "to_epoch": 9
  },
  {
    "delta": 0.007020220109691,
    "from_epoch": 9,
    "to_epoch": 10
  },
  {
    "delta": 0.008641007227793152,
    "from_epoch": 10,
    "to_epoch": 11
  },
  {
    "delta": 0.010528530801557134,
    "from_epoch": 11,
    "to_epoch": 12
  },
  {
    "delta": 0.021147375828199422,
    "from_epoch": 12,
    "to_epoch": 13
  },
  {
    "delta": 0.01446109067440432,
    "from_epoch": 13,
    "to_epoch": 14
  },
  {
    "delta": 0.02293513967998939,
    "from_epoch": 14,
    "to_epoch": 15
  },
  {
    "delta": 0.027236134522318156,
    "from_epoch": 15,
    "to_epoch": 16
  },
  {
    "delta": 0.027389979032211298,
    "from_epoch": 16,
    "to_epoch": 17
  },
  {
    "delta": 0.022252484394975625,
    "from_epoch": 17,
    "to_epoch": 18
  },
  {
    "delta": 0.03859725888229093,
    "from_epoch": 18,
    "to_epoch": 19
  },
  {
    "delta": 0.021905569106547895,
    "from_epoch": 19,
    "to_epoch": 20
  }
]
```

## Post-best later declines

```json
[]
```

## Overfitting intervals

The following intervals have decreasing train loss and increasing validation loss.

```json
[
  {
    "from_epoch": 4,
    "to_epoch": 5,
    "train_loss_delta": -0.003775093102209859,
    "validation_loss_delta": 0.0007516794942339367
  },
  {
    "from_epoch": 5,
    "to_epoch": 6,
    "train_loss_delta": -0.003873853025524676,
    "validation_loss_delta": 0.0005200263066091892
  },
  {
    "from_epoch": 6,
    "to_epoch": 7,
    "train_loss_delta": -0.004944946538862738,
    "validation_loss_delta": 0.00035089442815455785
  },
  {
    "from_epoch": 7,
    "to_epoch": 8,
    "train_loss_delta": -0.005286349137427593,
    "validation_loss_delta": 0.0022280439886248615
  },
  {
    "from_epoch": 8,
    "to_epoch": 9,
    "train_loss_delta": -0.006580628379650788,
    "validation_loss_delta": 0.002210697589299704
  },
  {
    "from_epoch": 9,
    "to_epoch": 10,
    "train_loss_delta": -0.00800994295365004,
    "validation_loss_delta": 0.007020220109691
  },
  {
    "from_epoch": 10,
    "to_epoch": 11,
    "train_loss_delta": -0.009410401991610473,
    "validation_loss_delta": 0.008641007227793152
  },
  {
    "from_epoch": 11,
    "to_epoch": 12,
    "train_loss_delta": -0.01193142129012309,
    "validation_loss_delta": 0.010528530801557134
  },
  {
    "from_epoch": 12,
    "to_epoch": 13,
    "train_loss_delta": -0.013182392021726302,
    "validation_loss_delta": 0.021147375828199422
  },
  {
    "from_epoch": 13,
    "to_epoch": 14,
    "train_loss_delta": -0.014649163575764956,
    "validation_loss_delta": 0.01446109067440432
  },
  {
    "from_epoch": 14,
    "to_epoch": 15,
    "train_loss_delta": -0.015078591748533071,
    "validation_loss_delta": 0.02293513967998939
  },
  {
    "from_epoch": 15,
    "to_epoch": 16,
    "train_loss_delta": -0.014832554405218024,
    "validation_loss_delta": 0.027236134522318156
  },
  {
    "from_epoch": 16,
    "to_epoch": 17,
    "train_loss_delta": -0.0142043869513592,
    "validation_loss_delta": 0.027389979032211298
  },
  {
    "from_epoch": 17,
    "to_epoch": 18,
    "train_loss_delta": -0.01378015453473,
    "validation_loss_delta": 0.022252484394975625
  },
  {
    "from_epoch": 18,
    "to_epoch": 19,
    "train_loss_delta": -0.012465622314755731,
    "validation_loss_delta": 0.03859725888229093
  },
  {
    "from_epoch": 19,
    "to_epoch": 20,
    "train_loss_delta": -0.011434807851308071,
    "validation_loss_delta": 0.021905569106547895
  }
]
```

## Existing early-stopped trajectory comparison

```json
[
  {
    "epoch": 1,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 2,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 3,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 4,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 5,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 6,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 7,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  },
  {
    "epoch": 8,
    "train_loss_delta_new_minus_baseline": 0.0,
    "validation_loss_delta_new_minus_baseline": 0.0
  }
]
```

## Artifacts

- `trajectory.csv`: epochs 1–20
- `trajectory_analysis.json`: machine-readable analysis
- `train_validation_loss_curve.png`: `{'created': True, 'path': '/workspace/project/analysis/stage2_encoder_decoder_5class_weighted_no_early_stop_20ep_v0/train_validation_loss_curve.png'}`
- `best.pt`: lowest validation weighted CE checkpoint
- `last.pt`: epoch 20 checkpoint
