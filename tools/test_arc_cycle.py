import json
import math
import re
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from arc_cycle import cycles, integral


class ArcTests(unittest.TestCase):
    def test_speed_integral_with_cut_between_samples(self):
        sec = [0, 0.8, 2.3, 4]
        velocity = [0, 1.6, 4.6, 8]  # v(t) = 2t, metres per second.
        upper = integral(sec, velocity, 0.2, 1.37)
        lower = integral(sec, velocity, 1.37, 3.7)
        self.assertAlmostEqual(upper, 1.37**2 - 0.2**2)
        self.assertAlmostEqual(upper + lower, 3.7**2 - 0.2**2)

    def fixture(self):
        return dict(sec=[0, 1, 2, 3, 4], spd=[36] * 5,
                    yb=[0, 1, 0, -1, 0], wp=[1] * 5,
                    turns=[dict(t0=1, t1=3, tflip=0.5, tfl=2,
                                L=dict(dyaw=-180), R=dict(dyaw=-180)),
                           dict(t0=3, t1=4, tflip=3.5, tfl=4,
                                L=dict(dyaw=180), R=dict(dyaw=180))])

    def test_uses_both_flip_points_instead_of_algorithm_cuts(self):
        a = cycles(self.fixture())[0]
        self.assertTrue(a['valid'])
        self.assertEqual((a['upper_distance'], a['lower_distance']), (15, 15))
        self.assertEqual(a['upper_duration'] + a['lower_duration'], 3)

    def test_missing_next_flip_is_not_replaced_by_turn_end(self):
        d = self.fixture()
        d['turns'][1]['tflip'] = None
        a = cycles(d)[0]
        self.assertFalse(a['valid'])
        self.assertIsNone(a['end'])
        self.assertNotIn('upper_distance', a)

    def test_fall_line_outside_cycle_is_not_clamped(self):
        d = self.fixture()
        d['turns'][1]['tflip'] = 1.5
        self.assertFalse(cycles(d)[0]['valid'])

    def test_repeated_timestamp_does_not_add_duration(self):
        self.assertAlmostEqual(integral([0, 1, 1, 2], [3, 3, 3, 3], 0, 2), 6)

    def test_missing_samples_are_not_treated_as_zero_speed(self):
        with self.assertRaises(ValueError):
            integral([0, 1, 2], [3, None, 3], 0, 2)


def audit():
    root = Path(__file__).resolve().parent.parent
    html = (root / 'index.html').read_text()
    start = re.search(r'\bALL\s*=', html).end()
    dataset = json.JSONDecoder().raw_decode(html[start:])[0]
    report = {}
    for k, d in dataset.items():
        c = json.loads((root / k / 'c4.json').read_text())
        arcs = cycles(d)
        count = Counter()
        for i, (t, a) in enumerate(zip(d['turns'], arcs)):
            if not a['valid']:
                count[a['reason']] += 1
                assert c['turns'][str(t['k'])]['转向分布'][0] == '?', (k, t['k'])
                continue
            count['valid'] += 1
            assert a['start'] == t['tflip']
            assert a['end'] == d['turns'][i + 1]['tflip']
            assert a['start'] < a['split'] < a['end']
            assert math.isclose(a['upper_duration'] + a['lower_duration'], a['end'] - a['start'], abs_tol=1e-9)
            full = integral(d['sec'], np.asarray(d['spd'], float) / 3.6, a['start'], a['end'])
            assert math.isclose(a['upper_distance'] + a['lower_distance'], full, abs_tol=1e-7), (k, t['k'])
            assert a['upper_distance'] >= 0 and a['lower_distance'] >= 0
        report[k] = dict(count)
    print('Dataset audit:', len(report), 'runs,', sum(sum(x.values()) for x in report.values()), 'turns')
    print('f3:', report['f3'])


if __name__ == '__main__':
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ArcTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    audit()
