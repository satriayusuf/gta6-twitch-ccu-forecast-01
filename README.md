# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/satriayusuf/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/satriayusuf/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch audience: given everything recorded up to the end of the release day's first UTC hour, it writes five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2 to 24, scored by pinball loss on the log scale. Entered in make.fun's [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty.

## How it works

A big console launch day is mostly the same wave replaying itself: New Zealand unlocks first, then Europe, then the UK, then the US coasts, and the Twitch audience follows the sun. So instead of extrapolating the target's own first hour, the program learns the *shape* of a launch day from the case's comparables — past launches whose full hourly curves ship inside every case file.

Each comparable is reduced to its release-day log-ratio shape, `log(viewers(h) / viewers(1))`. The target is described by its "eve profile": the 24 hourly counts before day start, relative to its hour-1 print, which show how far the unlock wave had already rolled by 00:00 UTC. Three models then vote:

- **Model B** — a per-hour weighted ridge from eve features (five eve lags, launch size, viewers per channel) to the shape; its spread is the weighted quantiles of its own residuals, slightly widened.
- **Model D** — the same idea on a richer feature set (thirteen eve lags, channel growth into hour 1), with the spread taken from exact leave-one-out residuals under a normal law.
- **Model F** — a three-factor model of the donor shapes (weighted principal components); the target's factor scores are ridge-predicted from its eve features, and its spread also comes from leave-one-out residuals.

Donors are weighted by how closely their eve profile matches the target's, and launches whose day was dominated by one simultaneous global unlock (audience absent before day start, or a peak more than ten times hour 1) are filtered out when enough rolling-midnight launches remain — GTA VI unlocks at midnight local time, so its day should look like theirs. The three models' quantiles are averaged in log space.

Validity is treated as sacred — an invalid case is scored as persistence and forfeits any prize — so every layer is wrapped: if anything unexpected happens, the program falls back to the persistence forecast rather than crashing or emitting a malformed file, and every emitted row is checked against the output contract (five finite, positive, non-decreasing quantiles per hour).

The constants (ridge penalty, kernel width, factor count, spread multiplier) were tuned by leave-one-out experiments over the comparable launches themselves: each launch in turn played the target, forecast from the others. On the rolling-midnight launches that resemble this one, the ensemble's mean pinball loss is about 0.21, against about 0.29 for the persistence baseline on the same launches.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

with the case files (`<id>.in`) in `input/`. Python 3.13, standard library only, no network; deterministic — the same case always produces the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=gta6twitchccu&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
