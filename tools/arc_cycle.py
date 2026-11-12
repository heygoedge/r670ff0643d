"""Split a turn between its signed Yaw extrema at the estimated fall line."""
import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def window(sec, values, start, end):
    """Interpolate both boundaries so adjacent integrals share one cut point."""
    sec, values = np.asarray(sec, float), np.asarray(values, float)
    if np.any(np.diff(sec) < 0):
        raise ValueError('sample times go backwards')
    keep = np.r_[np.diff(sec) > 0, True]  # Last record at a repeated timestamp.
    sec, values = sec[keep], values[keep]
    if not sec[0] <= start < end <= sec[-1]:
        raise ValueError('interval outside recorded samples')
    inside = (sec > start) & (sec < end)
    x = np.r_[start, sec[inside], end]
    y = np.r_[np.interp(start, sec, values), values[inside], np.interp(end, sec, values)]
    if not np.isfinite(y).all():
        raise ValueError('missing samples inside interval')
    return x, y


def integral(sec, values, start, end):
    x, y = window(sec, values, start, end)
    return float(_trapezoid(y, x))


def cycles(D):
    """Use the primary pipeline's Yaw-extremum cuts, with opposite signs."""
    raw_sec = np.asarray(D['sec'], float)
    keep = np.r_[np.diff(raw_sec) > 0, True]
    sec = raw_sec[keep]
    speed = np.asarray(D['spd'], float)[keep] / 3.6
    yaw = np.degrees(np.unwrap(np.asarray(D['yb'], float)[keep]))
    relative = yaw - np.polyval(np.polyfit(sec - sec[0], yaw, 1), sec - sec[0])
    rate = np.convolve(np.gradient(yaw, sec), np.ones(5) / 5, 'same')
    wp = np.abs(np.asarray(D['wp'], float)[keep])
    turns = D['turns']
    result = []
    for n, t in enumerate(turns):
        start, split, end = t.get('t0'), t.get('tfl'), t.get('t1')
        arc = dict(start=start, split=split, end=end, valid=False, basis='yaw_peak_pair')
        result.append(arc)
        if start is None:
            arc['reason'] = '缺少起始 Yaw 峰值'
            continue
        if end is None:
            arc['reason'] = '缺少结束 Yaw 峰值，无法确定完整半弧区间'
            continue
        if not start < end:
            arc['reason'] = 'Yaw 峰值时间不递增'
            continue
        if start < sec[0] or end > sec[-1]:
            arc['reason'] = '两个 Yaw 峰值之间的传感器记录不完整'
            continue
        a, b = np.interp([start, end], sec, relative)
        arc.update(start_yaw=float(a), end_yaw=float(b),
                   start_kind='正峰' if a > 0 else '负峰', end_kind='正峰' if b > 0 else '负峰')
        if not t.get('tfl_valid', True):
            arc['reason'] = t.get('tfl_reason') or '本弯没有可靠的滚落线估计'
            continue
        if a * b >= 0:
            arc['reason'] = '起止 Yaw 未形成正、负峰值对'
            continue
        if split is None or not start < split < end:
            arc['reason'] = '滚落线不在两个 Yaw 峰值之间'
            continue
        direction = np.sign(np.interp(t['t1'], sec, yaw) - np.interp(t['t0'], sec, yaw)) or 1
        positive_rate = np.maximum(rate * direction, 0)
        try:
            upper = integral(sec, speed, start, split)
            lower = integral(sec, speed, split, end)
            turn_up = integral(sec, positive_rate, start, split)
            turn_lo = integral(sec, positive_rate, split, end)
            x, r = window(sec, positive_rate, start, end)
            steps = (r[:-1] + r[1:]) / 2 * np.diff(x)
            total = float(steps.sum())
            threshold_time = float(x[min(int(np.searchsorted(np.cumsum(steps), total * 0.15)) + 1, len(x) - 1)])
            straight = integral(sec, speed, start, threshold_time) if total > 0 else 0.0
        except ValueError as e:
            arc['reason'] = '半弧区间内速度或转向数据缺失'
            continue
        def radius(a, b):
            mask = (sec >= a) & (sec <= b) & (wp > 0.15) & np.isfinite(speed)
            return float(np.median(speed[mask] / wp[mask])) if mask.sum() >= 3 else None
        arc.update(valid=True, reason='', upper_duration=split - start,
                   lower_duration=end - split, upper_distance=upper,
                   lower_distance=lower, upper_turn=turn_up, lower_turn=turn_lo,
                   upper_radius=radius(start, split), lower_radius=radius(split, end),
                   initial_straight_distance=straight)
    return result
