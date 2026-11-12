"""Split a two-foot mean Roll zero-crossing interval at the fall line."""
import numpy as np
from roll_boundaries import boundaries
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


def half_at(arc, timestamp):
    """One shared half-open partition for half arcs and half turns."""
    if not arc['valid'] or timestamp is None or not arc['start'] <= timestamp < arc['end']:
        return None
    return 'upper' if timestamp < arc['split'] else 'lower'


def cycles(D):
    """Use adjacent alternating zero crossings of the two-foot Roll mean."""
    raw_sec = np.asarray(D['sec'], float)
    keep = np.r_[np.diff(raw_sec) > 0, True]
    sec = raw_sec[keep]
    speed = np.asarray(D['spd'], float)[keep] / 3.6
    yaw = np.degrees(np.unwrap(np.asarray(D['yb'], float)[keep]))
    rate = np.convolve(np.gradient(yaw, sec), np.ones(5) / 5, 'same')
    wp = np.abs(np.asarray(D['wp'], float)[keep])
    turns = D['turns']
    result = []
    for t, boundary in zip(turns, boundaries(D)):
        start, split, end = boundary['start'], t.get('tfl'), boundary['end']
        arc = dict(boundary, split=split, boundary_valid=boundary['valid'])
        result.append(arc)
        if not boundary['valid']:
            continue
        arc['valid'] = False
        if start < sec[0] or end > sec[-1]:
            arc['reason'] = 'Roll 均值过零区间的传感器记录不完整'
            continue
        if not t.get('tfl_valid', True):
            arc['reason'] = t.get('tfl_reason') or '本弯没有可靠的滚落线估计'
            continue
        if split is None or not start < split < end:
            arc['reason'] = '滚落线不在两个 Roll 均值过零点之间'
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
            tail_sec, tail_rate = window(sec, positive_rate, split, end)
            moving = np.flatnonzero(tail_rate > 20)
            tail_end = float(tail_sec[moving[-1]]) if len(moving) else None
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
                   initial_straight_distance=straight, tail_turn_end=tail_end,
                   tail_straight_duration=None if tail_end is None else end-tail_end)
    return result
