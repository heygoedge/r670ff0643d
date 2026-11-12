"""Split one flip-to-flip cycle at the page's existing fall-line marker."""
import numpy as np


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
    return float(np.trapz(y, x))


def cycles(D):
    """Return an auditable interval for every turn; never substitute a yaw cut."""
    raw_sec = np.asarray(D['sec'], float)
    keep = np.r_[np.diff(raw_sec) > 0, True]
    sec = raw_sec[keep]
    speed = np.asarray(D['spd'], float)[keep] / 3.6
    yaw = np.degrees(np.asarray(D['yb'], float)[keep])
    relative = yaw - np.polyval(np.polyfit(sec - sec[0], yaw, 1), sec - sec[0])
    rate = np.convolve(np.gradient(yaw, sec), np.ones(5) / 5, 'same')
    wp = np.abs(np.asarray(D['wp'], float)[keep])
    turns = D['turns']
    result = []
    for n, t in enumerate(turns):
        nxt = turns[n + 1] if n + 1 < len(turns) else {}
        start, split, end = t.get('tflip'), t.get('tfl'), nxt.get('tflip')
        arc = dict(start=start, split=split, end=end, valid=False)
        result.append(arc)
        if start is None:
            arc['reason'] = '缺少本次翻刃起点'
            continue
        if end is None:
            arc['reason'] = '缺少下一次翻刃起点，无法确定完整半弧区间'
            continue
        if not start < end:
            arc['reason'] = '相邻翻刃起点时间不递增'
            continue
        if start < sec[0] or end > sec[-1]:
            arc['reason'] = '两个翻刃起点之间的传感器记录不完整'
            continue
        if split is None or not start < split < end:
            arc['reason'] = '滚落线不在两个翻刃起点之间'
            continue
        # tfl falls back to the closest yaw sample when the board never crosses
        # the mean direction. That fallback cannot define two measured halves.
        old = relative[(sec >= t['t0']) & (sec <= t['t1'])]
        if len(old) < 2 or not old.min() < 0 < old.max():
            arc['reason'] = '本弯未越过滚落线参照，无法可靠切分上下半弧'
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
