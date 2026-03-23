from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-10


def _utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def _with_columns(df: pd.DataFrame, columns: dict[str, object]) -> pd.DataFrame:
    if not columns:
        return df
    return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)


def _pick_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(a, dtype=np.float64)
        .rolling(window, min_periods=window)
        .corr(pd.Series(b, dtype=np.float64))
        .to_numpy(dtype=np.float64)
    )


def _rolling_autocorr(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype=np.float64)
    return series.rolling(window, min_periods=window).corr(series.shift(1)).to_numpy(dtype=np.float64)


def _rolling_linear_slope(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2) + EPS
    for end in range(window, n + 1):
        y = values[end - window : end]
        if np.isnan(y).any():
            continue
        y_centered = y - y.mean()
        out[end - 1] = float(np.dot(x_centered, y_centered) / denom)
    return out


def feat_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else np.ones(len(df), dtype=np.float64)
    idx = _utc_index(df)
    days = idx.normalize()
    n = len(df)

    vpoc = np.full(n, np.nan, dtype=np.float64)
    vah = np.full(n, np.nan, dtype=np.float64)
    val = np.full(n, np.nan, dtype=np.float64)

    for day in pd.unique(days):
        day_idx = np.flatnonzero(days == day)
        if len(day_idx) < 2:
            continue
        bucket_size = max(float(np.nanmedian(c[day_idx]) * 0.0001), 0.05)
        profile: dict[int, float] = {}
        for i in day_idx:
            tp = float((h[i] + l[i] + c[i]) / 3.0)
            bucket = int(round(tp / bucket_size))
            profile[bucket] = profile.get(bucket, 0.0) + float(v[i])
            ordered = np.array(sorted(profile.items()), dtype=np.float64)
            keys = ordered[:, 0].astype(np.int64)
            vols = ordered[:, 1]
            prices = keys.astype(np.float64) * bucket_size
            poc_pos = int(np.argmax(vols))
            total_vol = float(vols.sum())
            target = total_vol * 0.70
            lo_pos = hi_pos = poc_pos
            included = float(vols[poc_pos])
            while included < target and (lo_pos > 0 or hi_pos < len(vols) - 1):
                lo_add = float(vols[lo_pos - 1]) if lo_pos > 0 else -1.0
                hi_add = float(vols[hi_pos + 1]) if hi_pos < len(vols) - 1 else -1.0
                if lo_add >= hi_add and lo_pos > 0:
                    lo_pos -= 1
                    included += max(lo_add, 0.0)
                elif hi_pos < len(vols) - 1:
                    hi_pos += 1
                    included += max(hi_add, 0.0)
                else:
                    break
            vpoc[i] = prices[poc_pos]
            vah[i] = prices[hi_pos]
            val[i] = prices[lo_pos]

    cols = {
        "vpoc": vpoc,
        "vah": vah,
        "val": val,
        "vpoc_dist": (c - vpoc) / (c + EPS) * 100.0,
        "above_vah": (c > vah).astype(float),
        "below_val": (c < val).astype(float),
        "inside_va": ((c >= val) & (c <= vah)).astype(float),
        "dist_vah": (vah - c) / (c + EPS) * 100.0,
        "dist_val": (c - val) / (c + EPS) * 100.0,
    }
    return _with_columns(df, cols)


def feat_auction_theory(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    n = len(df)
    cols: dict[str, object] = {}

    for p in (10, 20, 50):
        time_above = np.full(n, np.nan, dtype=np.float64)
        time_below = np.full(n, np.nan, dtype=np.float64)
        if n > p:
            windows = np.lib.stride_tricks.sliding_window_view(c, p + 1)
            prev = windows[:, :-1]
            current = windows[:, -1][:, None]
            above = (prev > current).mean(axis=1)
            below = (prev < current).mean(axis=1)
            time_above[p:] = above
            time_below[p:] = below
        cols[f"time_above_{p}"] = time_above
        cols[f"time_below_{p}"] = time_below
        cols[f"price_acceptance_{p}"] = (np.abs(np.nan_to_num(time_above, nan=0.5) - 0.5) < 0.2).astype(float)
        cols[f"price_rejection_{p}"] = (np.abs(np.nan_to_num(time_above, nan=0.5) - 0.5) > 0.4).astype(float)

    daily_range = h - l
    prev_range = np.roll(daily_range, 1)
    prev_range[0] = daily_range[0]
    cols["range_extension"] = (daily_range > prev_range).astype(float)
    cols["range_contraction"] = (daily_range < prev_range * 0.7).astype(float)
    cols["range_ratio"] = daily_range / (prev_range + EPS)
    return _with_columns(df, cols)


def feat_tick_rule(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else np.ones(len(df), dtype=np.float64)
    n = len(df)

    tick_sign = np.sign(np.diff(c, prepend=c[0]))
    for i in range(1, n):
        if tick_sign[i] == 0:
            tick_sign[i] = tick_sign[i - 1]
    signed_vol = tick_sign * v

    cols: dict[str, object] = {
        "tick_sign": tick_sign.astype(np.float64),
        "signed_vol_lr": signed_vol,
    }

    signed_series = pd.Series(signed_vol, dtype=np.float64)
    vol_series = pd.Series(v, dtype=np.float64)
    for p in (10, 20, 50, 200):
        cum_sv = signed_series.rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
        cum_v = vol_series.rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
        cols[f"cum_signed_vol_{p}"] = np.nan_to_num(cum_sv)
        cols[f"flow_imbalance_{p}"] = np.nan_to_num(cum_sv / (cum_v + EPS))

    consec = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        consec[i] = consec[i - 1] + 1 if tick_sign[i] == tick_sign[i - 1] else 1
    cols["tick_consec"] = consec * tick_sign
    return _with_columns(df, cols)


def feat_return_distribution(df: pd.DataFrame) -> pd.DataFrame:
    c = np.clip(df["close"].to_numpy(dtype=np.float64), EPS, None)
    log_ret = np.concatenate([[0.0], np.diff(np.log(c))])
    ret_series = pd.Series(log_ret, dtype=np.float64)
    cols: dict[str, object] = {}

    for p in (20, 60, 240):
        m1 = ret_series.rolling(p, min_periods=p).mean()
        m2 = (ret_series**2).rolling(p, min_periods=p).mean()
        m3 = (ret_series**3).rolling(p, min_periods=p).mean()
        m4 = (ret_series**4).rolling(p, min_periods=p).mean()
        mu2 = np.clip(m2 - m1**2, 0.0, None)
        mu3 = m3 - 3.0 * m1 * m2 + 2.0 * (m1**3)
        mu4 = m4 - 4.0 * m1 * m3 + 6.0 * (m1**2) * m2 - 3.0 * (m1**4)
        skew = (mu3 / np.power(mu2 + EPS, 1.5)).to_numpy(dtype=np.float64)
        kurt = (mu4 / np.power(mu2 + EPS, 2.0) - 3.0).to_numpy(dtype=np.float64)
        autocorr = _rolling_autocorr(log_ret, p)
        cols[f"ret_skew_{p}"] = skew
        cols[f"ret_kurt_{p}"] = kurt
        cols[f"ret_autocorr_{p}"] = autocorr
        cols[f"mean_reverting_{p}"] = (np.nan_to_num(autocorr) < -0.1).astype(float)
        cols[f"trending_autocorr_{p}"] = (np.nan_to_num(autocorr) > 0.1).astype(float)
        cols[f"fat_tails_{p}"] = (np.nan_to_num(kurt) > 1.0).astype(float)

    return _with_columns(df, cols)


def feat_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    c = np.clip(df["close"].to_numpy(dtype=np.float64), EPS, None)
    log_ret = np.concatenate([[0.0], np.diff(np.log(c))])
    idx = _utc_index(df)
    hour = idx.hour
    day_of_week = idx.dayofweek
    ret_series = pd.Series(log_ret, index=df.index, dtype=np.float64)

    hourly_mean = ret_series.groupby(hour).transform(lambda s: s.expanding().mean().shift(1))
    hourly_std = ret_series.groupby(hour).transform(lambda s: s.expanding().std(ddof=0).shift(1))
    daily_mean = ret_series.groupby(day_of_week).transform(lambda s: s.expanding().mean().shift(1))
    overall_mean = ret_series.expanding().mean().shift(1)
    overall_std = ret_series.expanding().std(ddof=0).shift(1)

    cols = {
        "hourly_seasonal_ret": hourly_mean.to_numpy(dtype=np.float64),
        "hourly_seasonal_std": hourly_std.fillna(ret_series.expanding().std(ddof=0).shift(1)).to_numpy(dtype=np.float64),
        "daily_seasonal_ret": daily_mean.to_numpy(dtype=np.float64),
        "above_seasonal_avg": (
            hourly_mean.fillna(0.0) > overall_mean.fillna(0.0) + overall_std.fillna(0.0) * 0.5
        ).astype(float),
        "below_seasonal_avg": (
            hourly_mean.fillna(0.0) < overall_mean.fillna(0.0) - overall_std.fillna(0.0) * 0.5
        ).astype(float),
        "fri_close_window": ((day_of_week == 4) & (hour >= 19)).astype(int),
        "monday_open": ((day_of_week == 0) & (hour <= 2)).astype(int),
        "witching_hour": (((hour == 8) & (day_of_week < 5)) | ((hour == 13) & (day_of_week < 5)) | ((hour == 21) & (day_of_week == 4))).astype(int),
    }
    return _with_columns(df, cols)


def feat_hhll_streak(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    n = len(df)
    hh_streak = np.zeros(n, dtype=np.float64)
    ll_streak = np.zeros(n, dtype=np.float64)
    hl_streak = np.zeros(n, dtype=np.float64)
    lh_streak = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        hh_streak[i] = hh_streak[i - 1] + 1 if h[i] > h[i - 1] else 0
        ll_streak[i] = ll_streak[i - 1] + 1 if l[i] < l[i - 1] else 0
        hl_streak[i] = hl_streak[i - 1] + 1 if l[i] > l[i - 1] else 0
        lh_streak[i] = lh_streak[i - 1] + 1 if h[i] < h[i - 1] else 0

    cols = {
        "hh_streak": hh_streak,
        "ll_streak": ll_streak,
        "hl_streak": hl_streak,
        "lh_streak": lh_streak,
        "bull_streak": np.minimum(hh_streak + hl_streak, 20.0),
        "bear_streak": np.minimum(ll_streak + lh_streak, 20.0),
        "hh_exhaustion": ((hh_streak == 0) & (np.roll(hh_streak, 1) >= 5)).astype(float),
        "ll_exhaustion": ((ll_streak == 0) & (np.roll(ll_streak, 1) >= 5)).astype(float),
    }
    cols["hh_exhaustion"][0] = 0.0
    cols["ll_exhaustion"][0] = 0.0
    return _with_columns(df, cols)


def feat_macro_proxies(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    gold_ret = np.append(0.0, np.diff(c) / (c[:-1] + EPS))
    cols: dict[str, object] = {}

    spx_col = _pick_column(df, "spx500_h1_close", "us500_h1_close", "spx500_m5_close")
    dxy_col = _pick_column(df, "usdx_m5_close", "usdx_h1_close", "usdx_m15_close")
    ag_col = _pick_column(df, "xagusd_m5_close", "xagusd_m15_close", "xagusd_h1_close")

    if spx_col is not None:
        spx = np.nan_to_num(df[spx_col].to_numpy(dtype=np.float64))
        spx_ret = np.append(0.0, np.diff(spx) / (spx[:-1] + EPS))
        cols["gold_spx_corr_20"] = _rolling_corr(gold_ret, spx_ret, 20)
        cols["risk_on_signal"] = ((gold_ret > 0) & (spx_ret > 0)).astype(float)
        cols["risk_off_signal"] = ((gold_ret > 0) & (spx_ret < 0)).astype(float)

    if dxy_col is not None:
        dxy = np.nan_to_num(df[dxy_col].to_numpy(dtype=np.float64))
        dxy_ret = np.append(0.0, np.diff(dxy) / (dxy[:-1] + EPS))
        cols["gold_dxy_corr_20"] = _rolling_corr(gold_ret, dxy_ret, 20)
        cols["gold_dxy_both_up"] = ((gold_ret > 0) & (dxy_ret > 0)).astype(float)
        for p in (20, 60):
            corr = _rolling_corr(gold_ret, dxy_ret, p)
            cols[f"dxy_decoupling_{p}"] = (np.nan_to_num(corr) > -0.2).astype(float)

    if ag_col is not None and spx_col is not None:
        silver = np.nan_to_num(df[ag_col].to_numpy(dtype=np.float64))
        silver_ret = np.append(0.0, np.diff(silver) / (silver[:-1] + EPS))
        cols["metals_rally"] = ((gold_ret > 0) & (silver_ret > 0)).astype(float)

    return _with_columns(df, cols)


def feat_fibonacci(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"].astype(np.float64)
    l = df["low"].astype(np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    fib_levels = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)
    cols: dict[str, object] = {}

    for p in (50, 100, 200):
        swing_h = h.rolling(p, min_periods=p).max().to_numpy(dtype=np.float64)
        swing_l = l.rolling(p, min_periods=p).min().to_numpy(dtype=np.float64)
        swing_range = swing_h - swing_l
        dist_stack = []
        for fib in fib_levels:
            fib_ret = swing_h - fib * swing_range
            dist = np.abs(c - fib_ret) / (c + EPS) * 100.0
            fib_key = str(fib).replace(".", "")
            cols[f"fib_{fib_key}_{p}"] = fib_ret
            cols[f"dist_fib_{fib_key}_{p}"] = dist
            cols[f"near_fib_{fib_key}_{p}"] = (dist < 0.15).astype(float)
            dist_stack.append(dist)
        dist_matrix = np.column_stack(dist_stack)
        closest = np.full(len(df), np.nan, dtype=np.float64)
        valid_rows = np.isfinite(dist_matrix).any(axis=1)
        if valid_rows.any():
            closest[valid_rows] = np.nanmin(dist_matrix[valid_rows], axis=1)
        cols[f"nearest_fib_dist_{p}"] = closest
        cols[f"at_any_fib_{p}"] = (closest < 0.15).astype(float)

    return _with_columns(df, cols)


def feat_wyckoff(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else np.ones(len(df), dtype=np.float64)
    price_range = h - l
    vol_series = pd.Series(v, dtype=np.float64)
    cols: dict[str, object] = {}

    for p in (20, 50):
        price_range_std = pd.Series(price_range, dtype=np.float64).rolling(p, min_periods=p).std(ddof=0).to_numpy(dtype=np.float64)
        vol_trend = _rolling_linear_slope(v, p)
        ret = np.full(len(df), np.nan, dtype=np.float64)
        ret[p:] = (c[p:] - c[:-p]) / (c[:-p] + EPS) * 100.0
        avg_vol = vol_series.rolling(p, min_periods=p).mean().to_numpy(dtype=np.float64)
        baseline_vol = np.nanmedian(v) + EPS
        effort_result = (avg_vol / baseline_vol) / (np.abs(ret) + EPS)

        price_range_std_s = pd.Series(price_range_std, dtype=np.float64)
        low_vol_q = price_range_std_s.expanding(min_periods=p).quantile(0.33).shift(1).to_numpy(dtype=np.float64)
        high_vol_q = price_range_std_s.expanding(min_periods=p).quantile(0.66).shift(1).to_numpy(dtype=np.float64)
        high_v_q = vol_series.expanding(min_periods=p).quantile(0.66).shift(1).to_numpy(dtype=np.float64)
        low_v_q = vol_series.expanding(min_periods=p).quantile(0.33).shift(1).to_numpy(dtype=np.float64)
        effort_s = pd.Series(effort_result, dtype=np.float64)
        effort_q = effort_s.expanding(min_periods=p).quantile(0.80).shift(1).to_numpy(dtype=np.float64)

        cols[f"wyckoff_accum_{p}"] = ((price_range_std < low_vol_q) & (v > high_v_q)).astype(float)
        cols[f"wyckoff_distrib_{p}"] = ((price_range_std > high_vol_q) & (v < low_v_q)).astype(float)
        cols[f"vol_trend_{p}"] = np.nan_to_num(vol_trend)
        cols[f"vol_rising_{p}"] = (np.nan_to_num(vol_trend) > 0).astype(float)
        cols[f"effort_result_{p}"] = np.nan_to_num(effort_result)
        cols[f"absorption_{p}"] = (np.nan_to_num(effort_result) > np.nan_to_num(effort_q, nan=np.inf)).astype(float)

    return _with_columns(df, cols)


def feat_entropy(df: pd.DataFrame) -> pd.DataFrame:
    c = np.clip(df["close"].to_numpy(dtype=np.float64), EPS, None)
    log_ret = np.concatenate([[0.0], np.diff(np.log(c))])
    diff_abs = np.abs(np.diff(c, prepend=c[0]))
    cols: dict[str, object] = {}

    for p in (20, 60, 240):
        entropy = np.full(len(df), np.nan, dtype=np.float64)
        for i in range(p, len(df)):
            r = log_ret[i - p : i]
            if np.std(r) < EPS:
                entropy[i] = 0.0
                continue
            counts, _ = np.histogram(r, bins=10)
            probs = counts.astype(np.float64) / (counts.sum() + EPS)
            probs = probs[probs > 0]
            entropy[i] = -np.sum(probs * np.log2(probs + EPS))

        direct_dist = np.full(len(df), np.nan, dtype=np.float64)
        direct_dist[p:] = np.abs(c[p:] - c[:-p])
        total_path = pd.Series(diff_abs, dtype=np.float64).rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
        efficiency = direct_dist / (total_path + EPS)
        entropy_q = (
            pd.Series(entropy, dtype=np.float64)
            .expanding(min_periods=max(p, 10))
            .quantile(0.33)
            .shift(1)
            .to_numpy(dtype=np.float64)
        )

        cols[f"entropy_{p}"] = entropy
        cols[f"efficiency_{p}"] = efficiency
        cols[f"low_entropy_{p}"] = (entropy < entropy_q).astype(float)
        cols[f"high_efficiency_{p}"] = (np.nan_to_num(efficiency) > 0.3).astype(float)
        cols[f"random_walk_{p}"] = (np.nan_to_num(efficiency) < 0.1).astype(float)

    hurst = df["hurst"].to_numpy(dtype=np.float64) if "hurst" in df.columns else np.full(len(df), 0.5, dtype=np.float64)
    cols["predictability_score"] = (
        np.nan_to_num(cols.get("low_entropy_60", np.zeros(len(df), dtype=np.float64)))
        + np.nan_to_num(cols.get("high_efficiency_60", np.zeros(len(df), dtype=np.float64)))
        + (np.nan_to_num(hurst) > 0.55).astype(float)
    ) / 3.0
    return _with_columns(df, cols)


def feat_price_memory(df: pd.DataFrame) -> pd.DataFrame:
    c = np.clip(df["close"].to_numpy(dtype=np.float64), EPS, None)
    cols: dict[str, object] = {}

    for zone_pct in (0.1, 0.25, 0.5):
        step = np.log1p(zone_pct / 100.0)
        last_seen: dict[int, int] = {}
        bars_since = np.full(len(df), 999.0, dtype=np.float64)
        for i, price in enumerate(c):
            key = int(round(np.log(price) / step))
            zone = price * zone_pct / 100.0
            best = 999.0
            for neighbor in range(key - 2, key + 3):
                j = last_seen.get(neighbor)
                if j is not None and abs(c[j] - price) <= zone:
                    best = min(best, float(i - j))
            bars_since[i] = best
            last_seen[key] = i
        key = str(zone_pct).replace(".", "")
        cols[f"bars_since_visit_{key}pct"] = np.minimum(bars_since, 500.0)
        cols[f"recently_visited_{key}pct"] = (bars_since < 20).astype(float)
        cols[f"fresh_level_{key}pct"] = (bars_since > 200).astype(float)

    return _with_columns(df, cols)


def feat_candle_sequences(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)

    body = np.abs(c - o)
    rng = h - l + EPS
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_h[0] = h[0]
    prev_l[0] = l[0]

    inside_bar = ((h < prev_h) & (l > prev_l)).astype(float)
    outside_bar = ((h > prev_h) & (l < prev_l)).astype(float)
    inside_bar[0] = 0.0
    outside_bar[0] = 0.0

    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    bull_pinbar = ((lower_wick > 2 * body) & (lower_wick > upper_wick)).astype(float)
    bear_pinbar = ((upper_wick > 2 * body) & (upper_wick > lower_wick)).astype(float)

    bull_candle = (c > o) & (body / rng > 0.6)
    bear_candle = (c < o) & (body / rng > 0.6)
    three_bulls = (bull_candle & np.roll(bull_candle, 1) & np.roll(bull_candle, 2)).astype(float)
    three_bears = (bear_candle & np.roll(bear_candle, 1) & np.roll(bear_candle, 2)).astype(float)
    three_bulls[:2] = 0.0
    three_bears[:2] = 0.0

    is_doji = body / rng < 0.1
    morning_star = ((np.roll(c, 2) < np.roll(o, 2)) & is_doji & (c > o)).astype(float)
    evening_star = ((np.roll(c, 2) > np.roll(o, 2)) & is_doji & (c < o)).astype(float)
    morning_star[:2] = 0.0
    evening_star[:2] = 0.0

    tweezer_top = ((np.abs(h - prev_h) < rng * 0.05) & (c < o)).astype(float)
    tweezer_bot = ((np.abs(l - prev_l) < rng * 0.05) & (c > o)).astype(float)
    tweezer_top[0] = 0.0
    tweezer_bot[0] = 0.0

    cols: dict[str, object] = {
        "inside_bar": inside_bar,
        "outside_bar": outside_bar,
        "bull_pinbar": bull_pinbar,
        "bear_pinbar": bear_pinbar,
        "three_bulls": three_bulls,
        "three_bears": three_bears,
        "morning_star": morning_star,
        "evening_star": evening_star,
        "tweezer_top": tweezer_top,
        "tweezer_bot": tweezer_bot,
    }

    all_patterns = inside_bar + outside_bar + bull_pinbar + bear_pinbar + morning_star + evening_star
    pattern_series = pd.Series(all_patterns, dtype=np.float64)
    for p in (10, 20):
        cols[f"pattern_density_{p}"] = pattern_series.rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)

    return _with_columns(df, cols)


def feat_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else np.ones(len(df), dtype=np.float64)
    idx = _utc_index(df)
    days = idx.normalize()
    tp = (h + l + c) / 3.0

    session_vwap = np.full(len(df), np.nan, dtype=np.float64)
    session_std = np.full(len(df), np.nan, dtype=np.float64)

    for day in pd.unique(days):
        day_idx = np.flatnonzero(days == day)
        cum_tpv = 0.0
        cum_tpv2 = 0.0
        cum_v = 0.0
        for i in day_idx:
            cum_tpv += tp[i] * v[i]
            cum_tpv2 += (tp[i] ** 2) * v[i]
            cum_v += v[i]
            if cum_v > 0:
                vwap_i = cum_tpv / cum_v
                var_i = max(0.0, cum_tpv2 / cum_v - vwap_i**2)
                session_vwap[i] = vwap_i
                session_std[i] = np.sqrt(var_i) if var_i > 0 else 1e-10

    vwap_dev = (c - session_vwap) / (session_std + EPS)
    cols = {
        "session_vwap": session_vwap,
        "session_vwap_std": session_std,
        "vwap_deviation": vwap_dev,
        "above_vwap": (c > session_vwap).astype(float),
        "below_vwap": (c < session_vwap).astype(float),
        "vwap_1sd_up": (vwap_dev > 1).astype(float),
        "vwap_2sd_up": (vwap_dev > 2).astype(float),
        "vwap_1sd_dn": (vwap_dev < -1).astype(float),
        "vwap_2sd_dn": (vwap_dev < -2).astype(float),
        "vwap_extended": (np.abs(np.nan_to_num(vwap_dev)) > 2).astype(float),
        "vwap_dist_pct": (c - session_vwap) / (c + EPS) * 100.0,
    }
    return _with_columns(df, cols)


def feat_variance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    c = np.clip(df["close"].to_numpy(dtype=np.float64), EPS, None)
    lr = np.concatenate([[0.0], np.diff(np.log(c))])
    lr_series = pd.Series(lr, dtype=np.float64)
    cols: dict[str, object] = {}

    for q in (2, 4, 8, 16):
        window = max(q * 4, 32)
        one_bar_var = lr_series.rolling(window, min_periods=window).var(ddof=0).to_numpy(dtype=np.float64)
        q_ret = lr_series.rolling(q, min_periods=q).sum()
        q_var = q_ret.rolling(window - q + 1, min_periods=window - q + 1).var(ddof=0).to_numpy(dtype=np.float64)
        vr = q_var / (q * one_bar_var + EPS)
        cols[f"vr_{q}"] = vr
        cols[f"momentum_regime_{q}"] = (np.nan_to_num(vr) > 1.1).astype(float)
        cols[f"meanrev_regime_{q}"] = (np.nan_to_num(vr) < 0.9).astype(float)

    return _with_columns(df, cols)


def feat_oi_proxy(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else np.ones(len(df), dtype=np.float64)
    price_dir = np.sign(np.diff(c, prepend=c[0]))
    vol_dir = np.sign(np.diff(v, prepend=v[0]))

    new_longs = ((price_dir > 0) & (vol_dir > 0)).astype(float)
    short_cover = ((price_dir > 0) & (vol_dir < 0)).astype(float)
    new_shorts = ((price_dir < 0) & (vol_dir > 0)).astype(float)
    long_exit = ((price_dir < 0) & (vol_dir < 0)).astype(float)

    cols: dict[str, object] = {
        "oi_new_longs": new_longs,
        "oi_short_cover": short_cover,
        "oi_new_shorts": new_shorts,
        "oi_long_exit": long_exit,
    }

    for p in (20, 50):
        bull_oi = pd.Series(new_longs + short_cover, dtype=np.float64).rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
        bear_oi = pd.Series(new_shorts + long_exit, dtype=np.float64).rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
        cols[f"oi_bull_pressure_{p}"] = bull_oi
        cols[f"oi_bear_pressure_{p}"] = bear_oi
        cols[f"oi_balance_{p}"] = (bull_oi - bear_oi) / (bull_oi + bear_oi + EPS)

    return _with_columns(df, cols)


def feat_london_fix(df: pd.DataFrame) -> pd.DataFrame:
    idx = _utc_index(df)
    hour = idx.hour
    minute = idx.minute

    am_fix_window = ((hour == 10) & (minute >= 15) & (minute <= 45)).astype(int)
    pm_fix_window = (((hour == 14) & (minute >= 45)) | ((hour == 15) & (minute <= 15))).astype(int)
    am_pre_fix = ((hour == 10) & (minute < 15)).astype(int)
    pm_pre_fix = ((hour == 14) & (minute >= 30) & (minute < 45)).astype(int)
    am_post_fix = (((hour == 10) & (minute > 45)) | ((hour == 11) & (minute <= 30))).astype(int)

    mins = hour * 60 + minute
    am_fix_mins = 10 * 60 + 30
    pm_fix_mins = 15 * 60
    dist_am = (am_fix_mins - mins) % (24 * 60)
    dist_pm = (pm_fix_mins - mins) % (24 * 60)

    cols = {
        "am_fix_window": am_fix_window,
        "pm_fix_window": pm_fix_window,
        "am_pre_fix": am_pre_fix,
        "pm_pre_fix": pm_pre_fix,
        "am_post_fix": am_post_fix,
        "any_fix_window": np.maximum(am_fix_window, pm_fix_window),
        "fix_day_count": (am_fix_window | pm_fix_window).astype(int),
        "mins_to_am_fix": np.minimum(dist_am, 240),
        "mins_to_pm_fix": np.minimum(dist_pm, 240),
        "mins_to_fix": np.minimum(np.minimum(dist_am, 240), np.minimum(dist_pm, 240)),
    }
    return _with_columns(df, cols)


def feat_range_distribution(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    idx = _utc_index(df)
    days = idx.normalize()

    day_high_so_far = np.full(len(df), np.nan, dtype=np.float64)
    day_low_so_far = np.full(len(df), np.nan, dtype=np.float64)
    day_range_so_far = np.full(len(df), np.nan, dtype=np.float64)
    day_range_pct = np.full(len(df), np.nan, dtype=np.float64)
    completed_ranges: list[float] = []

    for day in pd.unique(days):
        day_idx = np.flatnonzero(days == day)
        adr = np.mean(completed_ranges[-20:]) if completed_ranges else np.nan
        cum_high = h[day_idx[0]]
        cum_low = l[day_idx[0]]
        for i in day_idx:
            cum_high = max(cum_high, h[i])
            cum_low = min(cum_low, l[i])
            current_range = cum_high - cum_low
            day_high_so_far[i] = cum_high
            day_low_so_far[i] = cum_low
            day_range_so_far[i] = current_range
            if np.isfinite(adr):
                day_range_pct[i] = current_range / (adr + EPS)
        completed_ranges.append(float(np.max(h[day_idx]) - np.min(l[day_idx])))

    rng = day_high_so_far - day_low_so_far + EPS
    cols = {
        "day_high_so_far": day_high_so_far,
        "day_low_so_far": day_low_so_far,
        "day_range_so_far": day_range_so_far,
        "day_range_pct_adr": day_range_pct,
        "range_used_50pct": (np.nan_to_num(day_range_pct) > 0.5).astype(float),
        "range_used_80pct": (np.nan_to_num(day_range_pct) > 0.8).astype(float),
        "range_used_100pct": (np.nan_to_num(day_range_pct) > 1.0).astype(float),
        "pos_in_day_range": (c - day_low_so_far) / rng,
    }
    return _with_columns(df, cols)


def feat_intermarket_divergence(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    gold_ret = np.append(0.0, np.diff(c) / (c[:-1] + EPS))
    cols: dict[str, object] = {}
    pairs = (
        ("xagusd", ("xagusd_m5_close", "xagusd_m15_close", "xagusd_h1_close")),
        ("eurusd", ("eurusd_m5_close", "eurusd_m15_close", "eurusd_h1_close")),
        ("usdx", ("usdx_m5_close", "usdx_m15_close", "usdx_h1_close")),
        ("spx500", ("spx500_h1_close", "us500_h1_close", "spx500_m5_close")),
    )

    for name, candidates in pairs:
        col = _pick_column(df, *candidates)
        if col is None:
            continue
        other = np.nan_to_num(df[col].to_numpy(dtype=np.float64))
        other_ret = np.append(0.0, np.diff(other) / (other[:-1] + EPS))
        gold_series = pd.Series(gold_ret, dtype=np.float64)
        other_series = pd.Series(other_ret, dtype=np.float64)
        for p in (5, 20):
            gold_mom = gold_series.rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
            other_mom = other_series.rolling(p, min_periods=p).sum().to_numpy(dtype=np.float64)
            divergence = np.sign(np.nan_to_num(gold_mom)) != np.sign(np.nan_to_num(other_mom))
            cols[f"{name}_diverge_{p}"] = divergence.astype(float)
            cols[f"{name}_agree_{p}"] = (~divergence).astype(float)
            cols[f"{name}_mom_diff_{p}"] = gold_mom - other_mom

    return _with_columns(df, cols)


def feat_smc_score(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    bull_score = np.zeros(n, dtype=np.float64)
    bear_score = np.zeros(n, dtype=np.float64)

    bull_components = (
        "bos_bullish",
        "bull_fvg",
        "bull_liq_sweep",
        "bull_ob",
        "bull_pinbar",
        "morning_star",
        "above_asian_high",
        "above_pdh",
        "mtf_full_bull",
    )
    bear_components = (
        "bos_bearish",
        "bear_fvg",
        "bear_liq_sweep",
        "bear_ob",
        "bear_pinbar",
        "evening_star",
        "below_asian_low",
        "below_pdl",
        "mtf_full_bear",
    )

    for comp in bull_components:
        if comp in df.columns:
            bull_score += np.nan_to_num(df[comp].to_numpy(dtype=np.float64))
    for comp in bear_components:
        if comp in df.columns:
            bear_score += np.nan_to_num(df[comp].to_numpy(dtype=np.float64))

    max_score = max(float(np.max(bull_score)), float(np.max(bear_score)), 1.0)
    cols = {
        "smc_bull_score": bull_score,
        "smc_bear_score": bear_score,
        "smc_net_score": bull_score - bear_score,
        "smc_high_conviction_bull": (bull_score >= 3).astype(float),
        "smc_high_conviction_bear": (bear_score >= 3).astype(float),
        "smc_score_normalized": (bull_score - bear_score) / (max_score + EPS),
    }
    return _with_columns(df, cols)


def feat_label_quality(df: pd.DataFrame) -> pd.DataFrame:
    cols: dict[str, object] = {}
    n = len(df)
    for horizon in (5, 15, 60):
        column = f"future_ret_{horizon}"
        if column not in df.columns:
            continue
        fut_ret = np.nan_to_num(df[column].to_numpy(dtype=np.float64))
        magnitude = np.zeros(n, dtype=np.float64)
        magnitude[np.abs(fut_ret) > 0.05] = 1.0
        magnitude[np.abs(fut_ret) > 0.15] = 2.0
        magnitude[np.abs(fut_ret) > 0.30] = 3.0
        cols[f"conviction_{horizon}"] = np.abs(fut_ret)
        cols[f"easy_long_{horizon}"] = (fut_ret > 0.15).astype(float)
        cols[f"easy_short_{horizon}"] = (fut_ret < -0.15).astype(float)
        cols[f"hard_trade_{horizon}"] = (np.abs(fut_ret) < 0.05).astype(float)
        cols[f"move_magnitude_{horizon}"] = magnitude
    return _with_columns(df, cols)


def add_more_high_value_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    df = feat_volume_profile(df)
    df = feat_auction_theory(df)
    df = feat_tick_rule(df)
    df = feat_return_distribution(df)
    df = feat_seasonality(df)
    df = feat_hhll_streak(df)
    df = feat_macro_proxies(df)
    df = feat_fibonacci(df)
    df = feat_wyckoff(df)
    df = feat_entropy(df)
    df = feat_price_memory(df)
    df = feat_candle_sequences(df)
    df = feat_session_vwap(df)
    df = feat_variance_ratio(df)
    df = feat_oi_proxy(df)
    df = feat_london_fix(df)
    df = feat_range_distribution(df)
    df = feat_intermarket_divergence(df)
    df = feat_smc_score(df)
    df = feat_label_quality(df)
    return df
