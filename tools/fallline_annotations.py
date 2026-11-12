"""Apply explicit video fall-line annotations before deriving turn metrics."""
import json
import re
from pathlib import Path


def parse_timecode(value):
    match = re.fullmatch(r'(\d{2}):(\d{2}):(\d{2})[:.](\d{2})', value)
    if match is None:
        raise ValueError('expected HH:MM:SS:cc or HH:MM:SS.cc')
    h, m, s, hundredths = map(int, match.groups())
    if h > 23 or m > 59 or s > 59:
        raise ValueError('timecode outside clock range')
    return h * 3600 + m * 60 + s + hundredths / 100


def load_annotations(path='data/fallline_annotations.json'):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {'runs': {}}


def apply_turn(key, turn, watermark_offset, annotations):
    entry = annotations.get('runs', {}).get(key, {}).get(str(turn['k']))
    if entry is None:
        return False
    when = round(parse_timecode(entry['video_time']) + watermark_offset, 3)
    if not turn['t0'] <= when <= turn['t1']:
        raise ValueError(f'{key} T{turn["k"]}: annotation outside recorded turn')
    if 'tfl_auto' not in turn:
        turn['tfl_auto'] = turn['tfl']
    turn.update(tfl=when, tfl_source='manual_video', tfl_video=entry['video_time'])
    for foot in ('L', 'R'):
        if foot in turn:
            edge = turn[foot].get('tec')
            turn[foot]['ec2fl'] = None if edge is None else round(when - edge, 2)
    return True


def apply_annotations(key, data, watermark_offset, annotations):
    return [turn['k'] for turn in data['turns']
            if apply_turn(key, turn, watermark_offset, annotations)]
