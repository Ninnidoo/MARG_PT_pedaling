# Run A Short Analysis

Run A completed 10 epochs; epoch 8 is the canonical best checkpoint by Frozen-PT direction-aware Transition F1.

| epoch | train loss | Human F1 | Frozen P | Frozen R | Frozen F1 | hard state agreement | first-MAIN agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.39446 | 0.5613 | 0.6981 | 0.3585 | 0.4737 | 0.4925 | 0.5714 |
| 2 | 1.21727 | 0.5791 | 0.7058 | 0.3733 | 0.4883 | 0.4936 | 0.5000 |
| 3 | 1.17656 | 0.5961 | 0.5716 | 0.5624 | 0.5670 | 0.4999 | 0.5714 |
| 4 | 1.13668 | 0.6148 | 0.6951 | 0.4276 | 0.5295 | 0.5107 | 0.6571 |
| 5 | 1.09995 | 0.6209 | 0.6841 | 0.4425 | 0.5374 | 0.5358 | 0.5714 |
| 6 | 1.06430 | 0.6240 | 0.5725 | 0.5855 | 0.5789 | 0.4977 | 0.5429 |
| 7 | 1.03518 | 0.6320 | 0.6051 | 0.5471 | 0.5747 | 0.5159 | 0.5286 |
| 8 | 1.01152 | 0.6380 | 0.5901 | 0.5772 | 0.5836 | 0.4888 | 0.5286 |
| 9 | 0.99471 | 0.6363 | 0.6223 | 0.5357 | 0.5758 | 0.5198 | 0.6429 |
| 10 | 0.98559 | 0.6436 | 0.6393 | 0.5239 | 0.5759 | 0.4906 | 0.6571 |

## Epoch 8

- Full PRE/MAIN/final-tail/POST horizon: P=0.590084, R=0.577186, F1=0.583564; predicted/reference ratio=0.978141.
- Common `[t1,tM)` horizon: P=0.589162, R=0.576268, F1=0.582644; candidate/reference=33697/34451.
- Count predictions N0/N1/N2: 39,489/4,868/901.
- Hard state agreement=0.488841; mismatch=0.511159; mismatch runs mean/median/p95/max=7.698/4.0/26.0/473.
- Recovery within 1/2/4/8 intervals=0.1297/0.2376/0.4019/0.6025.
- PRE synchronization=0.528571; first-MAIN agreement=0.528571; starting-ON PRE success=0.083333; starting-ON MAIN state agreement=0.511604.
- Frozen timing MAE is unavailable: 19 PT piece timelines pair to multiple human timings.

## Conclusion

The hypothesis is supported. Run A localized transitions substantially better than its early epochs and reached balanced epoch-8 P/R, but recurrent parity did not generalize as an absolute state trajectory: hard state agreement remained 0.489, near chance, and first-MAIN agreement was only 0.529. PRE cold-start synchronization did not generalize reliably, especially for starting-ON performances (0.083).

Epoch 9-10 Frozen F1 decreased mildly from 0.583564 to 0.575866 while train loss continued falling, consistent with mild over-specialization, not a collapse. No new inference or Stage 1 generation was performed; common-horizon evaluation reused epoch-8 candidate MIDI and the canonical 70 aligned references. ASAP test access: 0.
