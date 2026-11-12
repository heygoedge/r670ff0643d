import json
import unittest
from dataset import ROOT, load_dataset
from calibrate_fallline import calibrate, parse_timecode


class CalibrationTests(unittest.TestCase):
    def test_timecodes_use_hundredths_and_reject_invalid_clock(self):
        self.assertEqual(parse_timecode('12:10:50:92'),43850.92)
        with self.assertRaises(ValueError):parse_timecode('12:60:50:92')

    def test_four_labels_and_leave_one_out_remain_within_a_tenth_second(self):
        data,_=load_dataset();report=calibrate(data)
        self.assertEqual([s['turn']for s in report['samples']],[5,6,7,8])
        self.assertEqual(report['global_gain'],.65)
        self.assertLess(report['maximum_absolute_error_seconds'],.1)
        self.assertLess(report['leave_one_out_maximum_absolute_error_seconds'],.1)
        published=json.loads((ROOT/'tools/calibration-report.json').read_text())
        self.assertEqual(report,published)


if __name__=='__main__':unittest.main()
