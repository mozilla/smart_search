from datetime import datetime, timedelta
import math
from parser import parse_time_window


def get_half_life(start, end, edge_weight=0.25):
    """
    Compute the half life for an edge between two dates.
    by default weight decay to 25% at window edge
    """

    window_days = (end - start).total_seconds() / 86400.0
    half_span = window_days / 2.0
    # h = (half_span) / log2(1/edge_weight)
    h_days = half_span / (math.log(1.0/edge_weight, 2))
    return max(1.0, h_days)  # clamp to >= 1 day


def temporal_score(ts_ms, start, end):
    """
    Exponential decay by distance to window center.
    https://en.wikipedia.org/wiki/Exponential_decay
    """

    center = start + (end - start) / 2
    dist_sec = abs((datetime.fromtimestamp(ts_ms / 1000) - center).total_seconds())
    # half_life_sec = 2 * 24 * 3600
    half_life_sec = get_half_life(start, end) * 24 * 3600
    return math.exp(-math.log(2) * dist_sec / (half_life_sec + 1e-6))


def is_within_window(ts_ms, start, end):
    """
    (inside, different days)
    if inside=True:  (True, 0)
    if inside=False: diff days < 0 for before window, > 0 for after window
    """

    dt = datetime.fromtimestamp(ts_ms / 1000)
    if start <= dt <= end:
        return True, 0.0
    diff = (dt - (start if dt < start else end)).total_seconds() / 86400.0 # day
    return False, diff


def format_window_offset(inside, diff_days):
    """
    Format string for debug only.
    """
    if inside is None:
        return "-"
    if inside:
        return "in"
    sign = "+" if diff_days > 0 else ""
    return f"{sign}{diff_days:.1f} days"


def rank_results(histories, query, current_dt, alpha_temporal=0.7, lang="en", include_outside=False):
    """
    if extracts a window:
    score = alpha*temporal_score + (1-alpha)*frecency_norm
    else:
    score = alpha*recency_decay + (1-alpha)*frecency_norm
    """

    window = parse_time_window(query, current_dt, lang=lang)

    res = []

    if window:
        start, end = window
        for history in histories:
            last_visit_date = history.get("last_visit_date")
            frecency = history.get("frecency", 0.0)

            inside, diff_days = is_within_window(last_visit_date, start, end)

            if not include_outside and not inside:
                continue

            frecency_min, frecency_max = 100, 5000
            norm_frecency = (frecency - frecency_min) / (frecency_max - frecency_min)

            t_score = temporal_score(last_visit_date, start, end)
            # combine with normalized frecency score
            score = alpha_temporal * t_score + (1 - alpha_temporal) * frecency
            res.append({
                **history,
                "temporal_score": t_score,
                "score": score,
                "frecency_normalized": norm_frecency,
                "inside_window": inside,
                "window_diff_days": diff_days,
            })
    else:
        for history in histories:
            frecency = history.get("frecency", 0.0)
            frecency_min, frecency_max = 100, 5000
            norm_frecency = (frecency - frecency_min) / (frecency_max - frecency_min)
            res.append({
                **history,
                "temporal_score": 0.0,
                "score": norm_frecency,
                "frecency_normalized": norm_frecency,
                "inside_window": None,
                "window_diff_days": None,
            })

    res.sort(key=lambda x: x["score"], reverse=True)
    return res