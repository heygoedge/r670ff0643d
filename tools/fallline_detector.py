"""Estimate fall-line crossings from board Yaw and integrated route rotation.

The only fitted runtime parameter is ROUTE_GAIN. Video labels are test inputs,
never runtime overrides. Canonical precision matches the published sensor data.
"""
import numpy as np

ROUTE_GAIN = 0.65
VERSION = 'yaw-route-fusion-v1'


def prepare(sec, yaw_rad, route_rate_rad):
    sec = np.round(np.asarray(sec, float), 3)
    yaw = np.round(np.asarray(yaw_rad, float), 3)
    rate = np.round(np.asarray(route_rate_rad, float), 3)
    if len(sec) < 3 or np.any(np.diff(sec) < 0):
        raise ValueError('sensor times must be ordered and contain at least 3 samples')
    keep = np.r_[np.diff(sec) > 0, True]
    sec, yaw, rate = sec[keep], yaw[keep], rate[keep]
    if not (np.isfinite(sec).all() and np.isfinite(yaw).all() and np.isfinite(rate).all()):
        raise ValueError('non-finite heading or route samples')
    yaw = np.degrees(np.unwrap(yaw))
    relative = yaw - np.polyval(np.polyfit(sec - sec[0], yaw, 1), sec - sec[0])
    route = np.r_[0.0, np.cumsum((rate[:-1] + rate[1:]) / 2 * np.diff(sec))]
    return sec, relative, np.degrees(route)


def crossing(sec, heading, start, end, direction):
    inside = (sec > start) & (sec < end)
    x = np.r_[start, sec[inside], end]
    h = np.r_[np.interp(start, sec, heading), heading[inside], np.interp(end, sec, heading)]
    hit = np.flatnonzero((h[:-1] * h[1:] <= 0) & ((h[1:] - h[:-1]) * direction > 0))
    if not len(hit):
        return None
    times = x[hit] - h[hit] * (x[hit + 1] - x[hit]) / (h[hit + 1] - h[hit])
    return float(times[np.argmin(np.abs(times - (start + end) / 2))])


def detect(sec, yaw_rad, route_rate_rad, turns, gain=ROUTE_GAIN):
    sec, yaw, route = prepare(sec, yaw_rad, route_rate_rad)
    output = []
    for turn in turns:
        start, end = turn['t0'], turn['t1']
        a, b = np.interp([start, end], sec, yaw)
        direction = float(np.sign(b - a) or 1)
        # The route is anchored to the observed board heading at the first Yaw
        # extremum. Integration uses a_lateral / speed from the existing pipeline.
        route_heading = a + route - np.interp(start, sec, route)
        fused = yaw + gain * (route_heading - yaw)
        time = crossing(sec, fused, start, end, direction)
        method, valid = 'yaw_route_fusion', True
        reason = ''
        if time is None:
            time = crossing(sec, yaw, start, end, direction)
            method = 'yaw_fallback'
            reason = '路线融合方向未过零，回退到 Yaw 参照'
        if time is None:
            indices = np.flatnonzero((sec >= start) & (sec <= end))
            time = float(sec[indices[np.argmin(np.abs(yaw[indices]))]]) if len(indices) else start
            method, valid = 'unresolved', False
            reason = '本弯没有可靠的滚落线穿越，只显示最近参照点'
        output.append(dict(time=round(time, 3), method=method, valid=valid,
                           reason=reason, version=VERSION, gain=gain))
    return output


def apply(data, gain=ROUTE_GAIN):
    points = detect(data['sec'], data['yb'], data['wp'], data['turns'], gain)
    for turn, point in zip(data['turns'], points):
        for key in ('tfl_auto', 'tfl_video'):
            turn.pop(key, None)
        turn.update(tfl=point['time'], tfl_source=point['method'],
                    tfl_valid=point['valid'], tfl_reason=point['reason'],
                    tfl_version=point['version'])
        for foot in ('L', 'R'):
            if foot in turn:
                edge = turn[foot].get('tec')
                turn[foot]['ec2fl'] = None if edge is None else round(point['time'] - edge, 2)
    return points
