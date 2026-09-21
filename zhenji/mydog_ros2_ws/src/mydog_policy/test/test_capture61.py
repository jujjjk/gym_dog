import csv,json
import numpy as np
import pytest
from mydog_policy.capture61 import CaptureWriter

def test_writer_flushes_and_refuses_overwrite(tmp_path):
    w=CaptureWriter(tmp_path,{'schema':'test'})
    for i in range(100):w.submit(dict(control_step=i,obs_00=.1,raw_action_01=float('nan')))
    w.close()
    rows=list(csv.DictReader((tmp_path/'cycles.csv').open()))
    assert len(rows)==100 and rows[-1]['control_step']=='99'
    assert w.error=='' and w.dropped==0
    with pytest.raises(FileExistsError):CaptureWriter(tmp_path,{})

def test_passive_slice_copies_only_fresh_rows(tmp_path,monkeypatch):
    import time
    import sys
    from mydog_policy.capture61 import main
    source=tmp_path/'cycles.csv'
    now=time.monotonic()
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['timestamp_policy','policy_evaluated','obs_00'])
        w.writeheader()
        w.writerow(dict(timestamp_policy=now-5,policy_evaluated=False,obs_00=0))
        w.writerow(dict(timestamp_policy=now+.05,policy_evaluated=True,obs_00=.123))
    monkeypatch.setattr(sys,'argv',['slice','--source',str(source),'--label','trial01','--seconds','.1'])
    main()
    rows=list(csv.DictReader((tmp_path/'trial01.csv').open()))
    assert len(rows)==1 and float(rows[0]['obs_00'])==.123
    assert json.loads((tmp_path/'trial01.json').read_text())['policy_rows']==1
