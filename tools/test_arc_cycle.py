import math
import unittest
from collections import Counter

import numpy as np

from arc_cycle import cycles, integral
from dataset import ROOT, load_dataset
from fallline_detector import apply, detect, prepare


class ArcTests(unittest.TestCase):
    def fixture(self):
        return dict(sec=[0, 1, 2, 3, 4], spd=[36]*5, yb=[0, 1, 0, -1, 0], wp=[-.3]*5,
                    turns=[dict(k=1, t0=1, t1=3, tflip=.5, tfl=2),
                           dict(k=2, t0=3, t1=4, tflip=3.5, tfl=3.8)])

    def test_integral_has_exact_interpolated_boundaries(self):
        sec=[0,.8,2.3,4];velocity=[0,1.6,4.6,8]
        upper=integral(sec,velocity,.2,1.37);lower=integral(sec,velocity,1.37,3.7)
        self.assertAlmostEqual(upper,1.37**2-.2**2)
        self.assertAlmostEqual(upper+lower,3.7**2-.2**2)

    def test_half_arcs_use_yaw_peaks_and_ignore_flip_boundaries(self):
        d=self.fixture();a=cycles(d)[0]
        self.assertTrue(a['valid'])
        self.assertEqual((a['start'],a['end']),(1,3))
        self.assertEqual((a['upper_distance'],a['lower_distance']),(10,10))
        self.assertEqual((a['start_kind'],a['end_kind']),('正峰','负峰'))
        d['turns'][0]['tflip']=None;d['turns'][1]['tflip']=-100
        self.assertEqual(cycles(d)[0],a)

    def test_missing_peak_does_not_use_flip_point(self):
        d=self.fixture();d['turns'][0]['t1']=None
        self.assertFalse(cycles(d)[0]['valid'])
        self.assertNotIn('upper_distance',cycles(d)[0])

    def test_same_sign_boundaries_do_not_form_half_arcs(self):
        d=self.fixture();d['turns'][0].update(t0=.7,t1=1.3,tfl=1)
        self.assertFalse(cycles(d)[0]['valid'])

    def test_fall_line_outside_peaks_is_not_clamped(self):
        d=self.fixture();d['turns'][0]['tfl']=3.1
        self.assertFalse(cycles(d)[0]['valid'])

    def test_unresolved_fall_line_does_not_produce_a_ratio(self):
        d=self.fixture();d['turns'][0]['tfl_valid']=False
        self.assertFalse(cycles(d)[0]['valid'])

    def test_repeated_timestamp_does_not_add_duration(self):
        self.assertEqual(integral([0,1,1,2],[3,3,3,3],0,2),6)

    def test_missing_velocity_is_not_zero_velocity(self):
        with self.assertRaises(ValueError):integral([0,1,2],[3,None,3],0,2)

    def test_runtime_is_independent_of_turn_and_person_identifiers(self):
        d=self.fixture();a=apply(d)
        d['rider']='另一个滑手';d['key']='different'
        for t in d['turns']:t['k']+=100
        self.assertEqual(apply(d),a)

    def test_zero_gain_reduces_to_yaw_crossing(self):
        d=self.fixture();p=detect(d['sec'],d['yb'],d['wp'],d['turns'],gain=0)[0]
        self.assertEqual(p['time'],2)
        self.assertTrue(p['valid'])

    def test_route_gain_changes_the_estimate_in_the_expected_direction(self):
        d=self.fixture();a=detect(d['sec'],d['yb'],d['wp'],d['turns'],gain=0)[0]
        b=detect(d['sec'],d['yb'],d['wp'],d['turns'])[0]
        self.assertGreater(b['time'],a['time'])

    def test_legacy_manual_fields_cannot_override_inference(self):
        d=self.fixture();expected=apply(d)
        d['turns'][0].update(tfl=2.9,tfl_source='manual_video',tfl_video='00:00:02:90',tfl_auto=2)
        self.assertEqual(apply(d),expected)
        self.assertNotIn('tfl_video',d['turns'][0]);self.assertNotIn('tfl_auto',d['turns'][0])

    def test_duplicate_sensor_time_keeps_last_sample(self):
        sec,yaw,route=prepare([0,1,1,2],[0,9,1,0],[0,9,1,0])
        clean=prepare([0,1,2],[0,1,0],[0,1,0])
        for a,b in zip((sec,yaw,route),clean):np.testing.assert_array_equal(a,b)

    def test_sensor_time_must_not_go_backwards(self):
        with self.assertRaises(ValueError):prepare([0,2,1],[0,1,0],[0,1,0])


def audit():
    import json
    data,_=load_dataset();counts=Counter();invalid=Counter();rows=[]
    for key,d in data.items():
        c=json.loads((ROOT/key/'c4.json').read_text());pred=detect(d['sec'],d['yb'],d['wp'],d['turns'])
        for t,a,p in zip(d['turns'],cycles(d),pred):
            assert t['arc'].keys()==a.keys(),(key,t['k'],'embedded arc fields mismatch')
            for field,value in a.items():
                stored=t['arc'][field]
                assert (math.isclose(value,stored,abs_tol=1e-7,rel_tol=1e-9) if isinstance(value,float) else value==stored),(key,t['k'],field,'embedded arc mismatch')
            assert t['tfl']==p['time'] and t['tfl_source']==p['method'],(key,t['k'],'inference mismatch')
            assert 'tfl_video'not in t and 'tfl_auto'not in t
            counts['turns']+=1;counts[p['method']]+=1
            assert a['start']==t['t0']and a['end']==t['t1']
            assert t['t0']<=p['time']<=t['t1']
            if not a['valid']:
                invalid[a['reason']]+=1
                assert c['turns'][str(t['k'])]['转向分布'][0]=='?',(key,t['k'],'invalid half-arc has a conclusion')
            else:
                counts['valid_arcs']+=1
                assert a['start_yaw']*a['end_yaw']<0
                assert a['start']<a['split']<a['end']
                assert math.isclose(a['upper_duration']+a['lower_duration'],a['end']-a['start'],abs_tol=1e-9)
                full=integral(d['sec'],np.asarray(d['spd'],float)/3.6,a['start'],a['end'])
                assert math.isclose(a['upper_distance']+a['lower_distance'],full,abs_tol=1e-7),(key,t['k'])
                assert a['upper_distance']>=0 and a['lower_distance']>=0
            rows.append(dict(key=key,turn=t['k'],fall_line=p['time'],method=p['method'],valid_half_arc=a['valid'],
                             reason=a.get('reason',''),yaw_start=a['start'],yaw_end=a['end'],
                             upper_seconds=a.get('upper_duration'),lower_seconds=a.get('lower_duration'),
                             upper_metres=a.get('upper_distance'),lower_metres=a.get('lower_distance')))
    assert len(data)==18 and counts['turns']==587
    print('Dataset audit:',len(data),'runs',dict(counts),'invalid:',dict(invalid))
    return dict(runs=len(data),counts=dict(counts),invalid_reasons=dict(invalid),turns=rows)


if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ArcTests))
    if not result.wasSuccessful():raise SystemExit(1)
    audit()
