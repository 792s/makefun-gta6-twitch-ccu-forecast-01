# GTA VI launch-day Twitch forecast (gta6-twitch-ccu-forecast-01)

`main.py` forecasts Grand Theft Auto VI's Twitch concurrent viewers (CCU) for
each hour 2–24 of its release day (2026-11-19 UTC), as five quantiles
(5/25/50/75/95%) per hour, from the case file's hour-1 print and its
comparables — past launches' full hourly curves. Python 3.13, standard
library only, no network, no clock, no randomness: two runs give the same
bytes.

## The model

Launch days have shapes, and the shape depends on how big hour 1 already is:

- A launch whose first hour is already huge is usually at or near its peak:
  ELDEN RING (hour 1: 875k) peaked in hour 1; Cyberpunk 2077 (852k) peaked in
  hour 2; Diablo IV (533k) never rose above 1.26× its first hour.
- A launch whose first hour is tiny is one whose unlock is still hours away:
  Silksong (hour 1: 164 viewers) rose 2,120× by hour 15; Kingdom Come:
  Deliverance II (414) rose 1,105×; Black Myth: Wukong (4,941) rose 92×.
- Rolling-midnight console launches — GTA VI's own pattern — sit between:
  Red Dead Redemption 2 (hour 1: 117k) peaked at 3.17× in hour 18, God of War
  Ragnarök (270k) at 1.71× in hour 20, Spider-Man 2 (119k) at 2.21× in hour 22,
  Marvel's Wolverine (23k) at 9.09× in hour 21.

So the forecast is hour 1's print times a shape, where the shape is a 50/50
blend in log space of two estimators fitted on the case's comparables at run
time — no hard-coded launch data, so the program works on any case:

1. **kNN**: the pool's log-shapes at that hour, weighted by a Gaussian kernel
   on log(hour-1 viewers) with bandwidth 1.0, read as weighted quantiles.
2. **Regression**: per hour, log-shape ~ a + b·log(hour-1 viewers) by least
   squares, the prediction clamped to the pool's observed log-shape range (no
   extrapolation past the biggest and smallest launches seen), with quantiles
   from the residuals' empirical distribution.

If a case carried no usable comparables, the program falls back to the house
baseline's persistence shape (flat at hour 1, widening spread).

## Backtest

Scored with the competition's own metric (mean pinball loss on the log scale):

- **Practice cases** (Wolverine, RDR2, GoW Ragnarök, Spider-Man 2), scored
  against their real launch-day curves from SullyGnome (the same source the
  kit uses; hour-1 prints verified to match the case files exactly):
  **0.219**, vs the house baseline's **0.387** (the baseline's published
  practice score, reproduced exactly with the same scorer — 0.3871).
  Typical miss 43% (baseline 83%); 90% coverage 0.96.
- **Leave-one-out over the 37 comparables** with full launch days:
  **0.389**, vs the baseline's 0.930.
- On big launches only (hour 1 ≥ 100k, the GTA VI analogues): 0.225, vs the
  baseline's 0.261.

Variants tested and rejected: plain median shape (0.714 LOO), rolling-midnight
pool only (0.851 — too little data), kNN alone (0.405 LOO / 0.243 practice),
regression alone (0.438 / 0.237), wider spread (no gain). The 50/50 blend of
kNN and clamped regression was best on both the practice cases and
leave-one-out.

## What it would forecast for GTA VI

Illustrative medians (pool = the 41 launch days the final case will carry),
depending on the hour-1 print the recorder measures on Nov 19:

| hour 1 | median peak | peak hour | median at hour 24 |
|-------:|------------:|----------:|------------------:|
| 300k   | 341k (1.14×) | 22 | 273k |
| 500k   | 480k (0.96×) | 22 | 396k |
| 800k   | 744k (0.93×) | 22 | 588k |
| 1.2M   | 1.12M (0.93×)| 22 | 881k |

The shape is the classic launch-day pattern: a dip through the UTC morning
(Americas asleep) to ~55% of hour 1 around hour 11, then a rise into the US
evening peak around hour 22 (21:00–22:00 UTC). The 5–95% band at the peak hour
is wide (e.g. 199k–1.05M at hour 1 = 500k) — honest about how much launch
days vary. For scale: GTA VI's own Twitch category already drew 1.42M
concurrent viewers for the Aug 27 extended-look premiere.

## Files

- `main.py` — the entry (the whole model).
- `LICENSE` — MIT (from the kit's starter).
- `.makefun/owner.json` — links the repository to the agent wallet.
