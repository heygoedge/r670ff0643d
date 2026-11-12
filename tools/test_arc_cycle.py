import json
import math
import unittest
from collections import Counter
import numpy as np
from arc_cycle import cycles, integral, half_at
from dataset import ROOT, load_dataset
from fallline_detector import apply, detect, prepare
from roll_boundaries import mean_roll, lobes, boundaries


class ArcTests(unittest.TestCase):
    def fixture(self):
        return dict(sec=list(range(10)),spd=[36]*10,yb=[0,1,0,-1,0,1,0,-1,0,1],wp=[-.3]*10,
            rollL=[-2,-1,2,3,2,-2,-3,-1,2,3],rollR=[-4,1,2,1,-2,-4,-1,1,2,1],
            turns=[dict(k=1,e='后刃',t0=1.3,t1=3.8,tflip=.5,ec=1.7,tfl=2.5),
                   dict(k=2,e='前刃',t0=4.3,t1=6.8,tflip=3.5,ec=4.7,tfl=5.5)])

    def test_integral_interpolates_sample_gaps(self):
        sec=[0,.8,2.3,4];velocity=[0,1.6,4.6,8]
        a=integral(sec,velocity,.2,1.37);b=integral(sec,velocity,1.37,3.7)
        self.assertAlmostEqual(a,1.37**2-.2**2)
        self.assertAlmostEqual(a+b,3.7**2-.2**2)

    def test_mean_zero_boundaries_share_the_intermediate_cut(self):
        a,b=cycles(self.fixture())
        self.assertTrue(a['valid']and b['valid'])
        self.assertEqual((a['start'],a['end'],b['start'],b['end']),(1,4,4,7))
        self.assertEqual((a['upper_distance'],a['lower_distance']),(15,15))
        self.assertEqual((a['start_direction'],a['end_direction']),(1,-1))
        self.assertEqual((b['start_direction'],b['end_direction']),(-1,1))

    def test_zeroes_use_the_mean_curve_rather_than_one_foot(self):
        d=self.fixture();a=cycles(d)[0]
        self.assertEqual(a['start'],1)
        # At this instant neither individual Roll reading is zero.
        self.assertEqual((d['rollL'][1],d['rollR'][1]),(-1,1))
        self.assertEqual(mean_roll(d)[1],0)

    def test_flip_and_edge_establishment_do_not_change_roll_cuts(self):
        d=self.fixture();a=cycles(d)
        for t in d['turns']:t.update(tflip=None,ec=-100)
        self.assertEqual(cycles(d),a)

    def test_roll_boundaries_do_not_require_opposite_yaw_peaks(self):
        d=self.fixture();d['yb']=[1]*10
        self.assertTrue(all(a['valid']for a in cycles(d)))

    def test_zero_touch_without_sign_change_is_ignored(self):
        d=dict(sec=[0,1,2,3,4],rollL=[1,0,1,-1,1],rollR=[1,0,1,-1,1])
        full=[s for s in lobes(d)if s['start']is not None and s['end']is not None]
        self.assertEqual([(s['start'],s['end'])for s in full],[(2.5,3.5)])

    def test_zero_plateau_is_one_crossing_at_its_midpoint(self):
        d=dict(sec=list(range(7)),rollL=[1,0,0,-1,0,0,1],rollR=[1,0,0,-1,0,0,1])
        full=[s for s in lobes(d)if s['start']is not None and s['end']is not None]
        self.assertEqual([(s['start'],s['end'],s['sign'])for s in full],[(1.5,4.5,-1)])

    def test_missing_foot_data_does_not_bridge_a_crossing(self):
        d=dict(sec=[0,1,2,3,4],rollL=[1,None,-1,-1,1],rollR=[1,1,-1,-1,1])
        events={s['start']for s in lobes(d)if s['start']is not None}
        self.assertEqual(events,{3.5})

    def test_missing_zero_boundaries_do_not_use_yaw_or_edge_points(self):
        d=self.fixture();d['rollL']=[1]*10;d['rollR']=[1]*10
        a=cycles(d)[0]
        self.assertFalse(a['valid']);self.assertIsNone(a['start']);self.assertIsNone(a['end'])
        self.assertNotIn('upper_distance',a)

    def test_one_roll_lobe_cannot_be_counted_as_two_original_turns(self):
        d=self.fixture();d['turns'][1].update(e='后刃',t0=2,t1=3,tfl=2.6)
        for a in cycles(d):
            self.assertFalse(a['valid']);self.assertIn('多个原弯',a['reason'])

    def test_fall_line_outside_roll_interval_is_not_clamped(self):
        d=self.fixture();d['turns'][0]['tfl']=4.1
        self.assertFalse(cycles(d)[0]['valid'])

    def test_unresolved_fall_line_does_not_produce_half_metrics(self):
        d=self.fixture();d['turns'][0]['tfl_valid']=False
        self.assertFalse(cycles(d)[0]['valid'])

    def test_half_arcs_and_half_turns_share_one_partition(self):
        a=cycles(self.fixture())[0]
        for x,result in [(1,'upper'),(2.499,'upper'),(2.5,'lower'),(3.99,'lower'),(4,None),(.9,None)]:
            self.assertEqual(half_at(a,x),result)
        a['valid']=False;self.assertIsNone(half_at(a,2))

    def test_positive_rescaling_keeps_roll_zero_times(self):
        d=self.fixture();a=boundaries(d)
        for k in ('rollL','rollR'):d[k]=[2*v for v in d[k]]
        self.assertEqual(boundaries(d),a)

    def test_repeated_timestamp_does_not_add_duration(self):
        self.assertEqual(integral([0,1,1,2],[3,3,3,3],0,2),6)

    def test_missing_velocity_is_not_zero_velocity(self):
        with self.assertRaises(ValueError):integral([0,1,2],[3,None,3],0,2)

    def test_sensor_times_must_not_go_backwards(self):
        with self.assertRaises(ValueError):prepare([0,2,1],[0,1,0],[0,1,0])
        with self.assertRaises(ValueError):lobes(dict(sec=[0,2,1],rollL=[1,-1,1],rollR=[1,-1,1]))

    def test_fallline_inference_is_independent_of_identifiers(self):
        d=self.fixture();a=apply(d)
        d['rider']='另一个滑手';d['key']='different'
        for t in d['turns']:t['k']+=100
        self.assertEqual(apply(d),a)

    def test_manual_fields_cannot_override_fallline_inference(self):
        d=self.fixture();expected=apply(d)
        d['turns'][0].update(tfl=2.9,tfl_source='manual_video',tfl_video='00:00:02:90',tfl_auto=2)
        self.assertEqual(apply(d),expected)
        self.assertNotIn('tfl_video',d['turns'][0]);self.assertNotIn('tfl_auto',d['turns'][0])

    def test_duplicate_sensor_time_keeps_last_record(self):
        sec,yaw,route=prepare([0,1,1,2],[0,9,1,0],[0,9,1,0]);clean=prepare([0,1,2],[0,1,0],[0,1,0])
        for a,b in zip((sec,yaw,route),clean):np.testing.assert_array_equal(a,b)


def audit():
    data,_=load_dataset();counts=Counter();invalid=Counter();rows=[];max_zero_error=0
    for key,d in data.items():
        c=json.loads((ROOT/key/'c4.json').read_text());pred=detect(d['sec'],d['yb'],d['wp'],d['turns'])
        sec=np.array(d['sec'],float);keep=np.r_[np.diff(sec)>0,True];sec=sec[keep];mean=mean_roll(d)[keep]
        segments={p['id']:p for p in lobes(d)}
        np.testing.assert_allclose(d['rollMean'],mean_roll(d),atol=1e-10)
        for t,a,p in zip(d['turns'],cycles(d),pred):
            assert t['arc'].keys()==a.keys(),(key,t['k'],'embedded arc fields mismatch')
            for field,value in a.items():
                stored=t['arc'][field]
                assert (math.isclose(value,stored,abs_tol=1e-7,rel_tol=1e-9)if isinstance(value,float)else value==stored),(key,t['k'],field)
            assert a['basis']=='mean_roll_zero_pair'
            assert t['tfl']==p['time']and t['tfl_source']==p['method'],(key,t['k'],'fallline inference mismatch')
            assert 'tfl_video'not in t and 'tfl_auto'not in t
            counts['turns']+=1;counts[p['method']]+=1
            if a['boundary_valid']:counts['complete_roll_intervals']+=1
            if not a['valid']:
                invalid[a['reason']]+=1
                for slot in ('转向分布','峰值','分配','绊停'):
                    assert c['turns'][str(t['k'])][slot][0]=='?',(key,t['k'],slot,'invalid half interval has a conclusion')
            else:
                counts['valid_arcs']+=1
                lobe=segments[a['lobe_id']]
                assert (a['start'],a['end'])==(lobe['start'],lobe['end'])
                assert a['start_direction']==-a['end_direction']==lobe['sign']
                error=max(abs(float(np.interp(tm,sec,mean)))for tm in (a['start'],a['end']))
                max_zero_error=max(max_zero_error,error);assert error<1e-7,(key,t['k'],'boundary not on Roll mean zero',error)
                inner=mean[(sec>a['start'])&(sec<a['end'])]
                assert np.all(inner*lobe['sign']>=-1e-7),(key,t['k'],'skipped an intervening zero crossing')
                assert a['start']<a['split']<a['end']
                assert math.isclose(a['upper_duration']+a['lower_duration'],a['end']-a['start'],abs_tol=1e-9)
                full=integral(d['sec'],np.asarray(d['spd'],float)/3.6,a['start'],a['end'])
                assert math.isclose(a['upper_distance']+a['lower_distance'],full,abs_tol=1e-7),(key,t['k'])
                assert a['upper_distance']>=0 and a['lower_distance']>=0
                assert half_at(a,a['start'])=='upper'and half_at(a,a['split'])=='lower'and half_at(a,a['end'])is None
            rows.append(dict(key=key,turn=t['k'],fall_line=p['time'],method=p['method'],valid_half_arc=a['valid'],
                reason=a.get('reason',''),roll_zero_start=a['start'],roll_zero_end=a['end'],
                start_direction=a['start_direction'],end_direction=a['end_direction'],lobe_id=a['lobe_id'],
                upper_seconds=a.get('upper_duration'),lower_seconds=a.get('lower_duration'),
                upper_metres=a.get('upper_distance'),lower_metres=a.get('lower_distance')))
    assert len(data)==18 and counts['turns']==587
    print('Dataset audit:',len(data),'runs',dict(counts),'invalid:',dict(invalid),'maximum Roll boundary error:',max_zero_error)
    return dict(runs=len(data),counts=dict(counts),invalid_reasons=dict(invalid),maximum_roll_boundary_error_degrees=max_zero_error,turns=rows)


if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ArcTests))
    if not result.wasSuccessful():raise SystemExit(1)
    audit()
