"""Bounded, asynchronous capture writer and passive per-trial CSV slicer. No device I/O."""
import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path

class CaptureWriter:
    def __init__(self, directory, metadata):
        self.path=Path(directory)
        self.path.mkdir(parents=True,exist_ok=True)
        # Refuse overwrite of an earlier session.
        self.file=(self.path/'cycles.csv').open('x',newline='')
        (self.path/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        self.queue=queue.Queue(maxsize=500)
        self.dropped=0;self.written=0;self.error='';self.closed=False
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
    def submit(self,row):
        if self.closed or self.error:return
        try:self.queue.put_nowait(row)
        except queue.Full:self.dropped+=1
    def _run(self):
        writer=None
        try:
            while True:
                row=self.queue.get()
                if row is None:break
                if writer is None:
                    writer=csv.DictWriter(self.file,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);self.written+=1
                if self.written%25==0:self.file.flush()
        except Exception as exc:self.error=repr(exc)
        finally:self.file.flush();self.file.close()
    def close(self):
        self.closed=True
        while self.thread.is_alive():
            try:self.queue.put(None,timeout=.1);break
            except queue.Full:continue
        self.thread.join(timeout=10.)
        (self.path/'capture_summary.json').write_text(json.dumps(dict(
            written=self.written,dropped=self.dropped,error=self.error,
            writer_stopped=not self.thread.is_alive()),indent=2))

def main():
    p=argparse.ArgumentParser(description='Passive CSV capture: no ROS, motor or arm calls')
    p.add_argument('--source',required=True);p.add_argument('--label',required=True)
    p.add_argument('--seconds',type=float,default=30.)
    a=p.parse_args()
    if not 0<a.seconds<=180:p.error('seconds must be in (0,180]')
    if not a.label.replace('_','').replace('-','').isalnum():p.error('label must be alphanumeric, - or _')
    source=Path(a.source);start=time.monotonic();end=start+a.seconds
    # Wait for the source file before defining the actual window.
    if not source.is_file():p.error('cycles.csv does not exist; start capture controller first')
    print('Recording '+a.label+' (read-only), seconds='+str(a.seconds),flush=True)
    interrupted=False
    try:
        while time.monotonic()<end:time.sleep(max(0.,min(.1,end-time.monotonic())))
    except KeyboardInterrupt:end=time.monotonic();interrupted=True
    time.sleep(.6) # asynchronous writer flush
    dest=source.with_name(a.label+'.csv');count=0;policy=0;first=None;last=None
    with source.open() as f,dest.open('x',newline='') as g:
        reader=csv.DictReader(f);writer=csv.DictWriter(g,fieldnames=reader.fieldnames);writer.writeheader()
        for row in reader:
            try:t=float(row['timestamp_policy'])
            except (ValueError,TypeError):continue
            if start<=t<=end and None not in row and all(v is not None for v in row.values()):
                writer.writerow(row);count+=1;policy+=row['policy_evaluated']=='True'
                first=t if first is None else first;last=t
    meta=dict(label=a.label,window_monotonic=[start,end],rows=count,policy_rows=policy,
              first=first,last=last,interrupted=interrupted,source=str(source))
    dest.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    if count==0:raise SystemExit('No fresh samples captured; inspect controller and capture status')

if __name__=='__main__':main()
