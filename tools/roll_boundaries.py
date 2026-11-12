"""Adjacent alternating zero crossings of the displayed two-foot Roll mean."""
from collections import Counter
import numpy as np


def mean_roll(data):
    left=np.asarray(data['rollL'],float);right=np.asarray(data['rollR'],float)
    return np.round((left+right)/2,6)


def lobes(data):
    raw_sec=np.asarray(data['sec'],float)
    if len(raw_sec)<2 or np.any(np.diff(raw_sec)<0):raise ValueError('sensor times must be ordered')
    keep=np.r_[np.diff(raw_sec)>0,True]
    sec=raw_sec[keep];roll=mean_roll(data)[keep]
    finite=np.isfinite(sec)&np.isfinite(roll)
    edges=np.diff(np.r_[False,finite,False].astype(int))
    intervals=[]
    for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        nz=np.flatnonzero(roll[lo:hi]!=0)+lo
        if not len(nz):continue
        crossings=[]
        for i,j in zip(nz,nz[1:]):
            if roll[i]*roll[j]>=0:continue
            time=float((sec[i+1]+sec[j-1])/2) if j>i+1 else float(sec[i]-roll[i]*(sec[j]-sec[i])/(roll[j]-roll[i]))
            crossings.append((time,int(np.sign(roll[j]))))
        start=None;sign=int(np.sign(roll[nz[0]]))
        for end,direction in crossings+[(None,None)]:
            intervals.append(dict(id=len(intervals),start=start,end=end,sign=sign,
                                  coverage_start=float(sec[lo]) if start is None else start,
                                  coverage_end=float(sec[hi-1]) if end is None else end))
            start=end;sign=direction
    return intervals


def boundaries(data):
    """Keep existing turn IDs; use the same-edge lobe with greatest time overlap."""
    intervals=lobes(data);mapped=[]
    for turn in data['turns']:
        sign=1 if turn['e'] in ('后刃','heel') else -1
        candidates=[]
        for segment in intervals:
            overlap=min(turn['t1'],segment['coverage_end'])-max(turn['t0'],segment['coverage_start'])
            if segment['sign']==sign and overlap>0:
                distance=abs((segment['coverage_start']+segment['coverage_end']-turn['t0']-turn['t1'])/2)
                candidates.append((overlap,-distance,-segment['id'],segment))
        point=dict(start=None,end=None,basis='mean_roll_zero_pair',valid=False,
                   sign=sign,start_direction=sign,end_direction=-sign,lobe_id=None)
        mapped.append(point)
        if not candidates:
            point['reason']='本弯双脚 Roll 均值没有对应方向的正负区间'
            continue
        overlap,_,_,segment=max(candidates,key=lambda row:row[:3])
        point.update(start=segment['start'],end=segment['end'],lobe_id=segment['id'],
                     overlap_seconds=overlap,overlap_fraction=overlap/(turn['t1']-turn['t0']))
        if point['start']is None:point['reason']='缺少起始 Roll 均值过零点'
        elif point['end']is None:point['reason']='缺少结束 Roll 均值过零点'
        elif not point['start']<point['end']:point['reason']='Roll 均值过零时间不递增'
        else:point.update(valid=True,reason='')
    owners=Counter(point['lobe_id']for point in mapped if point['lobe_id']is not None)
    for point in mapped:
        if point['lobe_id']is not None and owners[point['lobe_id']]>1:
            point.update(valid=False,reason='同一 Roll 正负区间对应多个原弯，无法唯一划分')
    return mapped
