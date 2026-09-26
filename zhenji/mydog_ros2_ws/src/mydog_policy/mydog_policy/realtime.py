"""Best-effort Linux scheduling for the 50 Hz control process. Never fatal.

Measured on the Jetson (5.15 PREEMPT, schedutil): the control thread ran on
a core at 883 MHz of 1984 MHz and shared one interpreter lock with six
background threads, so 5 ms of policy work took 15-30 ms of wall time. The
kernel wake-up latency was under 2 ms. Nothing here changes motion; it only
asks the OS for a dedicated core and priority, and keeps the interpreter from
pausing the control thread for garbage collection or long GIL slices.

SCHED_FIFO needs an rtprio limit for the operator account and the clocks need
jetson_clocks; both are applied by jetson_realtime_setup.sh. Without them every
call below logs the refusal and control keeps running unchanged.
"""
import gc
import os
import sys
import threading

_lock = threading.Lock()
_reserved_tids = set()
_reports = {}


def parse_cpus(text):
    """'4' -> {4}; '0-3' -> {0,1,2,3}; '' -> None (leave affinity alone)."""
    cpus = set()
    for part in str(text or '').replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus or None


def configure_interpreter(switch_interval_sec=.001, gc_threshold=(50000, 20, 20)):
    """Shorter GIL slices bound how long a logging thread can hold the control
    thread off; larger gen0 threshold makes cyclic GC rare during control."""
    sys.setswitchinterval(float(switch_interval_sec))
    gc.set_threshold(*gc_threshold)


def freeze_startup_objects():
    """Move ONNX/ROS/config objects out of the collector's reach after setup so
    later full collections do not scan them in the control thread."""
    gc.collect()
    gc.freeze()


def apply_to_current_thread(role, cpus=None, fifo_priority=0, nice=None):
    """Pin/prioritize the calling thread. Returns a human-readable report."""
    tid = threading.get_native_id()
    notes = []
    with _lock:
        if fifo_priority:
            _reserved_tids.add(tid)
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
            notes.append('cpus=%s' % ','.join(str(c) for c in sorted(cpus)))
        except OSError as exc:
            notes.append('affinity refused (%s)' % exc.strerror)
    if fifo_priority:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(fifo_priority)))
            notes.append('SCHED_FIFO %d' % int(fifo_priority))
        except OSError as exc:
            notes.append('SCHED_FIFO refused (%s); run jetson_realtime_setup.sh' % exc.strerror)
    elif nice is not None:
        try:
            os.setpriority(os.PRIO_PROCESS, tid, int(nice))
            notes.append('nice %+d' % int(nice))
        except OSError as exc:
            notes.append('nice refused (%s)' % exc.strerror)
    report = ', '.join(notes) or 'unchanged'
    with _lock:
        _reports[role] = report
    return report


def confine_other_threads(cpus, nice=None):
    """Keep every non-reserved thread (IMU serial, motor poller, ROS executor,
    writers) off the control cores. Threads that later reserve a core override
    this for themselves."""
    if not cpus:
        return 'unchanged'
    moved = failed = 0
    with _lock:
        reserved = set(_reserved_tids)
    try:
        tids = [int(name) for name in os.listdir('/proc/self/task')]
    except OSError as exc:
        return 'task list unavailable (%s)' % exc.strerror
    for tid in tids:
        if tid in reserved:
            continue
        try:
            os.sched_setaffinity(tid, cpus)
            if nice is not None:
                os.setpriority(os.PRIO_PROCESS, tid, int(nice))
            moved += 1
        except OSError:
            failed += 1
    report = 'threads=%d on cpus=%s%s' % (
        moved, ','.join(str(c) for c in sorted(cpus)),
        '' if not failed else ', %d refused' % failed)
    with _lock:
        _reports['background'] = report
    return report


def reports():
    with _lock:
        return dict(_reports)
