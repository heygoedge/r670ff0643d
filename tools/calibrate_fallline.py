"""Fit one global gain; video labels are used only by this offline command."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from dataset import ROOT, load_dataset
from fallline_detector import ROUTE_GAIN, VERSION, crossing, detect, prepare

def parse_timecode(value):
    h,m,s,c=map(int,value.split(':'))
    if not(0<=h<24 and 0<=m<60 and 0<=s<60 and 0<=c<100):
        raise ValueError('invalid HH:MM:SS:hundredths')
    return h*3600+m*60+s+c/100

def calibrate(data):
    truth=json.loads((ROOT/'tools/fixtures/fallline-calibration.json').read_text())
    features={k:prepare(d['sec'],d['yb'],d['wp']) for k,d in data.items()}
    samples=[]
    for row in truth['samples']:
        t=next(t for t in data[row['key']]['turns']if t['k']==row['turn'])
        samples.append((row,t,parse_timecode(row['video_time'])+row['watermark_offset']))
    def predict(row,turn,gain):
        sec,yaw,route=features[row['key']];a,b=np.interp([turn['t0'],turn['t1']],sec,yaw)
        heading=yaw+gain*(a+route-np.interp(turn['t0'],sec,route)-yaw)
        return crossing(sec,heading,turn['t0'],turn['t1'],np.sign(b-a)or 1)
    def fit(points):
        def loss(gain):
            values=[predict(row,t,gain)for row,t,gt in points]
            return float(np.mean([(100 if value is None else (value-gt)**2)for value,(_,_,gt)in zip(values,points)]))
        return float(minimize_scalar(loss,bounds=(0,1),method='bounded').x)
    fitted=fit(samples);out=[]
    for i,(row,t,gt)in enumerate(samples):
        d=data[row['key']];point=detect(d['sec'],d['yb'],d['wp'],[t])[0]
        gain=fit(samples[:i]+samples[i+1:]);held=predict(row,t,gain)
        out.append(dict(**row,target_sensor_time=gt,predicted_sensor_time=point['time'],
                        error_seconds=round(point['time']-gt,6),leave_one_out_gain=gain,
                        leave_one_out_error_seconds=held-gt))
    return dict(version=VERSION,global_gain=ROUTE_GAIN,fitted_gain=fitted,samples=out,
                maximum_absolute_error_seconds=max(abs(x['error_seconds'])for x in out),
                mean_absolute_error_seconds=float(np.mean([abs(x['error_seconds'])for x in out])),
                leave_one_out_maximum_absolute_error_seconds=max(abs(x['leave_one_out_error_seconds'])for x in out),
                limitation='四个点来自同一位滑手，属于参数校准与逐点留出检查；不代表其他滑手或全部 587 弯的真实滚落线准确率。')

if __name__=='__main__':
    data,_=load_dataset();report=calibrate(data)
    path=ROOT/'tools/calibration-report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
