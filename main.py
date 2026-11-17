"""Launch-day Twitch viewers forecaster (make.fun: gta6-twitch-ccu-forecast-01).

Reads one case file (schema makefun/twitch-ccu-case/v1) per run from the input
directory and writes, for each of hours 2..24 of the launch's UTC release day,
five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent
viewers in the game's Twitch category.

Method, in one paragraph: a launch day is mostly the worldwide midnight-unlock
wave replaying itself, so the case's own comparables (past launches with full
hourly curves) carry the signal. Each comparable is reduced to its release-day
log-ratio shape, log(viewers(h) / viewers(1)). Three models turn the target's
pre-launch "eve" profile (the 24 hourly counts before day start, relative to
its hour-1 print) into a shape forecast: (B) a per-hour weighted ridge from eve
features to the shape, with the spread read off its in-sample residuals'
weighted quantiles; (D) the same ridge on a richer feature set, with the
spread from leave-one-out residuals (hat-matrix shortcut) under a normal law;
(F) a three-factor model of the donor shapes (weighted PCA), the target's
factor scores again ridge-predicted from its eve features, spread from
leave-one-out residuals. Donors are weighted by the distance between their eve
profile and the target's, and launches whose day was dominated by a single
global unlock moment (audience absent before day start, or a peak more than
ten times hour 1) are filtered out when enough rolling-midnight launches
remain. The three models' quantiles are averaged in log space. Every layer is
wrapped so that any failure falls back to the persistence forecast (hour 1's
print with a random-walk spread): this program must never crash or emit a
malformed file, because an invalid case forfeits the prize.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard
library only, no network). Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; put it back first
# so a sibling module would import (this entry is a single file, but the line
# costs nothing and keeps the layout standard).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))
LAM = 10.0            # ridge penalty (the intercept is never penalized)
EVE_SIGMA = 1.0       # width of the eve-profile distance kernel
NFACTORS = 3          # shape factors in model F
SCALE_B = 1.1         # spread multiplier of model B


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def wquantile(pairs, tau):
    """Weighted quantile of [(value, weight)] by linear interpolation."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    prev_v, prev_acc = pairs[0][0], 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            if acc == prev_acc:
                return v
            return prev_v + (target - prev_acc) / (acc - prev_acc) * (v - prev_v)
        prev_v, prev_acc = v, acc
    return pairs[-1][0]


def smooth(vals):
    """One pass of a [0.25, 0.5, 0.25] moving average, edges renormalized."""
    out = []
    n = len(vals)
    for i in range(n):
        acc = 0.0
        wsum = 0.0
        for j, kw in ((i - 1, 0.25), (i, 0.5), (i + 1, 0.25)):
            if 0 <= j < n:
                acc += kw * vals[j]
                wsum += kw
        out.append(acc / wsum)
    return out


def solve(A, b):
    """Solve a small dense linear system by Gauss-Jordan with pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / pv
                for cc in range(col, n + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def ridge_system(X, y, w, lam):
    """Weighted ridge fit; returns (beta, A_inv). Intercept unpenalized."""
    k = len(X[0])
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi, wi in zip(X, y, w):
        for a in range(k):
            b[a] += wi * xi[a] * yi
            for c in range(k):
                A[a][c] += wi * xi[a] * xi[c]
    for a in range(1, k):
        A[a][a] += lam
    cols = []
    for col in range(k):
        e = [1.0 if r == col else 0.0 for r in range(k)]
        cols.append(solve(A, e))
    Ainv = [[cols[c][r] for c in range(k)] for r in range(k)]
    beta = [sum(Ainv[a][c] * b[c] for c in range(k)) for a in range(k)]
    return beta, Ainv


def ridge_fit(X, y, w, lam):
    return ridge_system(X, y, w, lam)[0]


def top_eigvecs(cov, k, iters=200):
    """Top-k eigenvectors of a symmetric matrix (power iteration + deflation)."""
    n = len(cov)
    A = [row[:] for row in cov]
    vecs = []
    for _ in range(k):
        v = [1.0 + 0.01 * i for i in range(n)]
        for _ in range(iters):
            w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / norm for x in w]
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam = sum(v[i] * Av[i] for i in range(n))
        vecs.append(v)
        for i in range(n):
            for j in range(n):
                A[i][j] -= lam * v[i] * v[j]
    return vecs


# --------------------------------------------------------------------------
# Records: turning a case into a target record and donor records
# --------------------------------------------------------------------------

def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hourly_series(rows, value_key="viewers"):
    """Map from-timestamp -> value for rows that carry a usable number."""
    out = {}
    for row in rows or []:
        try:
            v = row.get(value_key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[parse_ts(row["from"])] = float(v)
        except Exception:
            continue
    return out


def day_and_eve(hourly, day_start):
    """Split an hourly map into release day (24 values, hours 1..24) and the
    24 eve hours before it (hours -24..-1); missing hours come back None."""
    day = [hourly.get(day_start + timedelta(hours=k)) for k in range(24)]
    eve = [hourly.get(day_start - timedelta(hours=k)) for k in range(24, 0, -1)]
    return day, eve


def build_records(case):
    """Returns (target, donors, hours). Donors keep only launches whose whole
    release day is present; the target record's day is a placeholder anchored
    at its hour-1 print (only index 0 is ever read for the target)."""
    day_start = parse_ts(case["day_start"])
    known = case.get("known") or []
    first = next((r for r in known if r.get("hour") == 1), None)
    hist = hourly_series(case.get("history_hourly"))
    hist_ch = hourly_series(case.get("history_hourly"), "channels")
    if first is not None and isinstance(first.get("viewers"), (int, float)):
        p1 = float(first["viewers"])
        ch1 = first.get("channels")
    elif hist:
        p1 = hist[max(hist)]
        ch1 = hist_ch.get(max(hist))
    else:
        p1, ch1 = 1.0, None
    p1 = max(1.0, p1)
    _, eve = day_and_eve(hist, day_start)
    _, ch_eve = day_and_eve(hist_ch, day_start)
    target = {
        "name": (case.get("game") or {}).get("name", ""),
        "day": [p1] + [p1] * 23,
        "eve": eve,
        "ch1": float(ch1) if isinstance(ch1, (int, float)) else None,
        "ch_eve": ch_eve,
    }
    donors = []
    for comp in case.get("comparables") or []:
        try:
            ds = parse_ts(comp["day_start"])
            hourly = hourly_series(comp.get("hourly"))
            chmap = hourly_series(comp.get("hourly"), "channels")
            day, dev_eve = day_and_eve(hourly, ds)
            _, dev_ch_eve = day_and_eve(chmap, ds)
            if any(v is None for v in day) or not day[0] or day[0] < 1:
                continue
            donors.append({
                "name": comp.get("name", ""),
                "day": day,
                "eve": dev_eve,
                "ch1": chmap.get(ds),
                "ch_eve": dev_ch_eve,
            })
        except Exception:
            continue
    hours = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if isinstance(h, int) and 2 <= h <= 24 and h not in hours:
            hours.append(h)
    hours = sorted(hours) or list(HOURS)
    return target, donors, hours


# --------------------------------------------------------------------------
# Shared model pieces
# --------------------------------------------------------------------------

def eve_profile(rec):
    """Log eve viewers relative to hour 1, for the last 12 eve hours."""
    p1 = max(1.0, rec["day"][0])
    return [math.log(max(1.0, v) / p1) if v else None for v in rec["eve"][12:]]


def eve_distance(pa, pb):
    ds = [(x - y) ** 2 for x, y in zip(pa, pb) if x is not None and y is not None]
    return math.sqrt(sum(ds) / len(ds)) if ds else 3.0


def regime_ok(day, eve):
    """A rolling-unlock-like day: the audience is already there before the day
    starts and no single later moment dwarfs hour 1 by more than tenfold."""
    p1 = max(1.0, day[0])
    if max(day) / p1 > 10.0:
        return False
    if not eve or eve[-1] is None or eve[-1] < 0.2 * p1:
        return False
    return True


def prepare_pool(target, donors):
    """The shared donor context every model reads: the filtered pool, the
    eve-kernel weights, the donors' log-ratio shapes and their mean log size.
    Computed once per case (the three models used to rebuild it each)."""
    pool = [d for d in donors if d["name"] != target["name"]]
    filt = [d for d in pool if regime_ok(d["day"], d["eve"])]
    if len(filt) >= 8:
        pool = filt
    if len(pool) < 4:
        raise ValueError("too few comparable launches")
    tp = eve_profile(target)
    weights = [math.exp(-(eve_distance(tp, eve_profile(d)) ** 2) / (2 * EVE_SIGMA ** 2)) for d in pool]
    ratios = [[math.log(max(1.0, d["day"][h - 1]) / max(1.0, d["day"][0])) for h in HOURS] for d in pool]
    mean_logp1 = sum(math.log(max(1.0, d["day"][0])) for d in pool) / len(pool)
    return pool, weights, ratios, mean_logp1


def features_small(rec, mean_logp1):
    """Model B features: five eve lags, size, viewers per channel."""
    p1 = max(1.0, rec["day"][0])
    feats = [1.0]
    for lag in (1, 4, 7, 10, 13):
        v = rec["eve"][-lag] if lag <= len(rec["eve"]) else None
        feats.append(math.log(max(1.0, v) / p1) if v else 0.0)
    feats.append(math.log(p1) - mean_logp1)
    feats.append(math.log(p1 / rec["ch1"]) if rec.get("ch1") else 0.0)
    return feats


def features_big(rec, mean_logp1):
    """Models D/F features: thirteen eve lags, size, viewers per channel,
    and the growth of the channel count into hour 1."""
    p1 = max(1.0, rec["day"][0])
    feats = [1.0]
    for lag in (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24):
        v = rec["eve"][-lag] if lag <= len(rec["eve"]) else None
        feats.append(math.log(max(1.0, v) / p1) if v else 0.0)
    feats.append(math.log(p1) - mean_logp1)
    feats.append(math.log(p1 / rec["ch1"]) if rec.get("ch1") else 0.0)
    ch_eve = rec.get("ch_eve") or []
    if rec.get("ch1") and ch_eve and ch_eve[-1]:
        feats.append(math.log(rec["ch1"] / ch_eve[-1]))
    else:
        feats.append(0.0)
    return feats


# --------------------------------------------------------------------------
# The three models. Each returns {hour: [q05, q25, q50, q75, q95]}.
# --------------------------------------------------------------------------

def model_b(target, donors, ctx=None):
    """Per-hour ridge on the small feature set; spread = weighted quantiles of
    its in-sample residuals, widened by SCALE_B."""
    pool, W, R, mean_logp1 = ctx if ctx is not None else prepare_pool(target, donors)
    p1 = max(1.0, target["day"][0])
    Xt = features_small(target, mean_logp1)
    X = [features_small(d, mean_logp1) for d in pool]
    n = len(pool)
    centers = []
    offs = {tau: [] for tau in LEVELS}
    for i in range(23):
        y = [R[j][i] for j in range(n)]
        beta = ridge_fit(X, y, W, LAM)
        centers.append(sum(b * x for b, x in zip(beta, Xt)))
        resid = [(y[j] - sum(b * x for b, x in zip(beta, X[j])), W[j]) for j in range(n)]
        for tau in LEVELS:
            offs[tau].append(wquantile(resid, tau))
    centers = smooth(centers)
    for tau in LEVELS:
        offs[tau] = smooth(offs[tau])
    out = {}
    for i, h in enumerate(HOURS):
        row = []
        for tau in LEVELS:
            off = offs[tau][i] * SCALE_B
            if tau < 0.5:
                off = min(off, 0.0)
            elif tau > 0.5:
                off = max(off, 0.0)
            row.append(p1 * math.exp(centers[i] + off))
        row.sort()
        out[h] = row
    return out


def loo_residuals(X, y, W, beta, Ainv):
    """Exact leave-one-out residuals of a weighted ridge fit, via leverages."""
    k = len(X[0])
    resid = []
    for j in range(len(X)):
        xj = X[j]
        quad = 0.0
        for a in range(k):
            s = 0.0
            for c in range(k):
                s += Ainv[a][c] * xj[c]
            quad += xj[a] * s
        hj = min(0.95, W[j] * quad)
        ej = y[j] - sum(b * x for b, x in zip(beta, xj))
        resid.append(ej / (1.0 - hj))
    return resid


def model_d(target, donors, ctx=None):
    """Per-hour ridge on the big feature set; spread = weighted standard
    deviation of its leave-one-out residuals, under a normal law."""
    pool, W, R, mean_logp1 = ctx if ctx is not None else prepare_pool(target, donors)
    p1 = max(1.0, target["day"][0])
    Xt = features_big(target, mean_logp1)
    X = [features_big(d, mean_logp1) for d in pool]
    n = len(pool)
    wsum = sum(W)
    centers = []
    sigmas = []
    for i in range(23):
        y = [R[j][i] for j in range(n)]
        beta, Ainv = ridge_system(X, y, W, LAM)
        centers.append(sum(b * x for b, x in zip(beta, Xt)))
        resid = loo_residuals(X, y, W, beta, Ainv)
        mu = sum(W[j] * resid[j] for j in range(n)) / wsum
        var = sum(W[j] * (resid[j] - mu) ** 2 for j in range(n)) / wsum
        sigmas.append(math.sqrt(max(var, 1e-4)))
    centers = smooth(centers)
    sigmas = smooth(sigmas)
    out = {}
    for i, h in enumerate(HOURS):
        row = [p1 * math.exp(centers[i] + z * sigmas[i]) for z in Z]
        row.sort()
        out[h] = row
    return out


def model_f(target, donors, ctx=None):
    """Factor model: the weighted mean shape plus three principal shape
    factors, the target's scores ridge-predicted from its eve features;
    spread from leave-one-out residuals of the factor fit."""
    pool, W, R, mean_logp1 = ctx if ctx is not None else prepare_pool(target, donors)
    p1 = max(1.0, target["day"][0])
    Xt = features_big(target, mean_logp1)
    X = [features_big(d, mean_logp1) for d in pool]
    n = len(pool)
    wsum = sum(W)
    mu_shape = [sum(W[j] * R[j][i] for j in range(n)) / wsum for i in range(23)]
    cov = [[0.0] * 23 for _ in range(23)]
    for j in range(n):
        dev = [R[j][i] - mu_shape[i] for i in range(23)]
        for a in range(23):
            for b in range(23):
                cov[a][b] += W[j] * dev[a] * dev[b]
    cov = [[c / wsum for c in row] for row in cov]
    vecs = top_eigvecs(cov, NFACTORS)
    scores = [[sum((R[j][i] - mu_shape[i]) * vecs[k][i] for i in range(23))
               for k in range(NFACTORS)] for j in range(n)]
    center = mu_shape[:]
    for k in range(NFACTORS):
        y = [scores[j][k] for j in range(n)]
        beta = ridge_fit(X, y, W, LAM)
        s = sum(b * x for b, x in zip(beta, Xt))
        for i in range(23):
            center[i] += s * vecs[k][i]
    # Leave-one-out residuals of the whole factor fit, for the spread.
    resid = [[0.0] * 23 for _ in range(n)]
    for j in range(n):
        idx = [m for m in range(n) if m != j]
        fit = mu_shape[:]
        for k in range(NFACTORS):
            y = [scores[m][k] for m in idx]
            beta = ridge_fit([X[m] for m in idx], y, [W[m] for m in idx], LAM)
            s = sum(b * x for b, x in zip(beta, X[j]))
            for i in range(23):
                fit[i] += s * vecs[k][i]
        for i in range(23):
            resid[j][i] = R[j][i] - fit[i]
    sigmas = []
    for i in range(23):
        mres = sum(W[j] * resid[j][i] for j in range(n)) / wsum
        var = sum(W[j] * (resid[j][i] - mres) ** 2 for j in range(n)) / wsum
        sigmas.append(math.sqrt(max(var, 1e-4)))
    center = smooth(center)
    sigmas = smooth(sigmas)
    out = {}
    for i, h in enumerate(HOURS):
        row = [p1 * math.exp(center[i] + z * sigmas[i]) for z in Z]
        row.sort()
        out[h] = row
    return out


# --------------------------------------------------------------------------
# Fallback, ensembling, output hygiene
# --------------------------------------------------------------------------

def persistence_rows(p1, hours):
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in hours}


def round2(x):
    """Round to 2 decimals, half up (as JavaScript's Math.round does);
    whole numbers come back as ints so the JSON stays compact."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def clean_rows(rows_by_hour, hours):
    """Force every row into the output contract: five finite numbers, each in
    (0, 1e9], non-decreasing, rounded to 2 decimals."""
    out = []
    for h in hours:
        row = list(rows_by_hour[h])
        fixed = []
        prev = 0.01
        for v in row:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                v = prev
            v = min(max(float(v), 0.01), 1e9)
            v = max(v, prev)
            fixed.append(round2(v))
            prev = fixed[-1]
        out.append({"hour": h, "q": fixed})
    return {"hours": out}


def forecast_case(case):
    target, donors, hours = build_records(case)
    p1 = max(1.0, target["day"][0])
    try:
        ctx = prepare_pool(target, donors)
    except Exception:
        ctx = None
    forecasts = []
    for model in (model_b, model_d, model_f):
        try:
            forecasts.append(model(target, donors, ctx))
        except Exception:
            continue
    if not forecasts:
        return clean_rows(persistence_rows(p1, hours), hours)
    merged = {}
    for h in hours:
        row = []
        for k in range(5):
            acc = 0.0
            for f in forecasts:
                acc += math.log(max(1e-9, f[h][k]))
            row.append(math.exp(acc / len(forecasts)))
        merged[h] = row
    return clean_rows(merged, hours)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            result = forecast_case(case)
        except Exception:
            # Last-resort output: a flat persistence file at one viewer, so the
            # case is a (hopeless but) well-formed file rather than a missing
            # one. Nothing in here may raise.
            try:
                result = clean_rows(persistence_rows(1.0, list(HOURS)), list(HOURS))
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
