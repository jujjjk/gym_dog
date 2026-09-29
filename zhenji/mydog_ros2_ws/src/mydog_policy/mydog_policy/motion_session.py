"""Command-session utilities; never opens a device or arms a controller."""
import numpy as np


class CommandOwner:
    def __init__(self, namespace):
        import fcntl
        self.file=open('/tmp/'+namespace.strip('/').replace('/','_')+'_command.lock','a')
        try: fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('Another command session is running; exit its B terminal first')
    def close(self): self.file.close()


class CommandRamp:
    """Bound command changes, not steady speed or total motion duration."""
    def __init__(self, caps):
        self.caps=np.asarray(caps,dtype=float)
        self.rates=self.caps/.5
        self.reset()
    def reset(self):self.value=np.zeros(3);self.time=None
    def update(self,target,now):
        target=np.asarray(target,dtype=float).reshape(3)
        if not np.isfinite(target).all() or np.any(np.abs(target)>self.caps+1e-8):
            raise ValueError('Command outside configured caps')
        dt=0. if self.time is None else np.clip(now-self.time,0.,.05)
        self.time=float(now)
        self.value+=np.clip(target-self.value,-self.rates*dt,self.rates*dt)
        return self.value.copy()


def actions(caps):
    x,y,w=map(float,caps);back=min(.3,x)
    return dict(w=(x,0,0),s=(-back,0,0),a=(0,y,0),d=(0,-y,0),
                m=(0,0,0),q=(0,0,w),e=(0,0,-w),
                forward_left=(min(.3,x),min(.15,y),0),
                forward_right=(min(.3,x),-min(.15,y),0),
                backward_left=(-back,min(.15,y),0),
                backward_right=(-back,-min(.15,y),0),
                combined=(min(.3,x),min(.15,y),min(.5,w)),
                combined_reverse=(-back,-min(.15,y),-min(.5,w)))


def ready(status, age):
    return bool(age<.2 and status.get('mode')=='ready' and status.get('walk_start_stable')
                and status.get('send') and not status.get('stand_only')
                and status.get('timing_ready') and status.get('observation_temporal_ok')
                and status.get('imu_calibrated') and not status.get('calibration_requested', False)
                and not status.get('trial_armed', False) and not status.get('walk_inhibit_latched'))
