#!/usr/bin/env python3
"""
gpu_monitor.py -- sample NVIDIA GPU usage to a CSV time series.

Reusable in two ways:

1. As a context manager around a workload (used by the Dorado sim):
       from gpu_monitor import GpuMonitor
       with GpuMonitor("gpu_usage.csv", interval=1.0, track_pids=[proc.pid]):
           run_workload()

2. As a standalone CLI that samples until interrupted, or for a fixed
   duration, or while a given PID is alive:
       python gpu_monitor.py --out gpu_usage.csv --interval 1
       python gpu_monitor.py --out gpu_usage.csv --pid 12345
       python gpu_monitor.py --out gpu_usage.csv --duration 60

Records per sample, per GPU: utilization (GPU + memory), memory used/total,
power draw, temperature, SM/memory clocks, fan speed, plus per-process
GPU memory and compute utilization for tracked PIDs (and their children).

Requires: nvidia-ml-py  (pip install nvidia-ml-py), and optionally psutil
(pip install psutil) to follow child processes of a tracked PID.
"""

import argparse
import csv
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
import pynvml


# Optional but recommended; to track children of a process 
try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# NVML sentinel for "value not available" on unsigned long long fields.
_NOT_AVAIL_ULL = (1 << 64) - 1


def _safe(call, *args, default=None):
    """Run an NVML call, returning `default` if it isn't supported / errors."""
    try:
        return call(*args)
    except pynvml.NVMLError:
        return default


def _decode(value):
    """NVML strings are sometimes bytes depending on binding version."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


CSV_FIELDS = [
    "timestamp_iso",
    "elapsed_s",
    "gpu_index",
    "gpu_name",
    "util_gpu_pct",
    "util_mem_pct",
    "mem_used_mib",
    "mem_total_mib",
    "power_w",
    "power_limit_w",
    "temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "fan_pct",
    "tracked_pids",            # ; separated PIDs found on this GPU
    "tracked_mem_used_mib",    # sum of GPU mem used by tracked PIDs on this GPU
    "tracked_util_sm_pct",     # sum of SM util attributed to tracked PIDs
]


class GpuMonitor:
    """
    Background sampler. Use as a context manager or call start()/stop().

    Parameters
    ----------
    out_path : str
        CSV file to write.
    interval : float
        Seconds between samples.
    track_pids : list[int] | None
        PIDs whose per-process usage should be isolated. Children are
        included automatically when psutil is available (Dorado may spawn
        worker processes).
    gpu_indices : list[int] | None
        Restrict to these GPU indices; None = all visible GPUs.
    """

    def __init__(self, out_path, interval=1.0, track_pids=None,
                 gpu_indices=None):
        self.out_path = out_path
        self.interval = max(0.05, float(interval))
        self.track_pids = list(track_pids) if track_pids else []
        self.gpu_indices = gpu_indices
        self._stop = threading.Event()
        self._thread = None
        self._start_wall = None
        self._handles = []
        self._sample_count = 0

    def start(self):
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        indices = (self.gpu_indices if self.gpu_indices is not None
                   else range(count))
        for i in indices:
            if i < 0 or i >= count:
                print(f"[gpu] WARNING: GPU index {i} out of range (0..{count-1})",
                      file=sys.stderr)
                continue
            self._handles.append((i, pynvml.nvmlDeviceGetHandleByIndex(i)))
        if not self._handles:
            pynvml.nvmlShutdown()
            raise RuntimeError("No GPUs to monitor.")

        self._start_wall = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        _safe(pynvml.nvmlShutdown)
        print(f"[gpu] wrote {self._sample_count} samples to {self.out_path}",
              file=sys.stderr)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # helpers
    def _expand_pids(self):
        """Return the tracked PID set, including live children."""
        pids = set(self.track_pids)
        if _HAVE_PSUTIL:
            for pid in list(self.track_pids):
                try:
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        pids.add(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return pids

    def _per_process(self, handle, tracked):
        """
        Return (pids_on_gpu, mem_used_mib, util_sm_pct) for tracked PIDs
        on a single GPU. Both compute and graphics process tables are
        consulted; utilization is best-effort (not all drivers support it).
        """
        found = []
        mem_bytes = 0
        # Memory via running-process tables.
        for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                   getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None)):
            if fn is None:
                continue
            procs = _safe(fn, handle, default=[]) or []
            for p in procs:
                if p.pid in tracked:
                    found.append(p.pid)
                    used = getattr(p, "usedGpuMemory", None)
                    if used is not None and used != _NOT_AVAIL_ULL:
                        mem_bytes += used

        # Per-process SM utilization (best effort). Looks back ~1s.
        util_sm = 0
        lookback_us = int((time.time() - self.interval - 1.0) * 1_000_000)
        samples = _safe(pynvml.nvmlDeviceGetProcessUtilization,
                        handle, lookback_us, default=None)
        if samples:
            for s in samples:
                if s.pid in tracked:
                    util_sm += int(getattr(s, "smUtil", 0) or 0)

        pids_unique = sorted(set(found))
        return pids_unique, mem_bytes / (1024 * 1024), util_sm

    def _sample_row(self, idx, handle, elapsed, tracked):
        name = _decode(_safe(pynvml.nvmlDeviceGetName, handle, default=""))

        util = _safe(pynvml.nvmlDeviceGetUtilizationRates, handle)
        util_gpu = util.gpu if util else ""
        util_mem = util.memory if util else ""

        mem = _safe(pynvml.nvmlDeviceGetMemoryInfo, handle)
        mem_used = round(mem.used / (1024 * 1024), 1) if mem else ""
        mem_total = round(mem.total / (1024 * 1024), 1) if mem else ""

        power = _safe(pynvml.nvmlDeviceGetPowerUsage, handle)
        power_w = round(power / 1000.0, 2) if power is not None else ""
        plimit = _safe(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        plimit_w = round(plimit / 1000.0, 2) if plimit is not None else ""

        temp = _safe(pynvml.nvmlDeviceGetTemperature, handle,
                     pynvml.NVML_TEMPERATURE_GPU)
        temp_c = temp if temp is not None else ""

        sm_clock = _safe(pynvml.nvmlDeviceGetClockInfo, handle,
                         pynvml.NVML_CLOCK_SM)
        mem_clock = _safe(pynvml.nvmlDeviceGetClockInfo, handle,
                          pynvml.NVML_CLOCK_MEM)
        fan = _safe(pynvml.nvmlDeviceGetFanSpeed, handle)

        pids, tmem, tutil = ([], "", "")
        if tracked:
            pids, tmem, tutil = self._per_process(handle, tracked)
            tmem = round(tmem, 1)

        return {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 3),
            "gpu_index": idx,
            "gpu_name": name,
            "util_gpu_pct": util_gpu,
            "util_mem_pct": util_mem,
            "mem_used_mib": mem_used,
            "mem_total_mib": mem_total,
            "power_w": power_w,
            "power_limit_w": plimit_w,
            "temp_c": temp_c,
            "sm_clock_mhz": sm_clock if sm_clock is not None else "",
            "mem_clock_mhz": mem_clock if mem_clock is not None else "",
            "fan_pct": fan if fan is not None else "",
            "tracked_pids": ";".join(str(p) for p in pids),
            "tracked_mem_used_mib": tmem,
            "tracked_util_sm_pct": tutil,
        }

    #  main loop
    def _run(self):
        new_file = not os.path.exists(self.out_path) or \
            os.path.getsize(self.out_path) == 0
        with open(self.out_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            while not self._stop.is_set():
                tick = time.time()
                elapsed = tick - self._start_wall
                tracked = self._expand_pids() if self.track_pids else set()
                for idx, handle in self._handles:
                    row = self._sample_row(idx, handle, elapsed, tracked)
                    writer.writerow(row)
                    self._sample_count += 1
                fh.flush()
                # Sleep the remainder of the interval, accounting for sample cost.
                drift = self.interval - (time.time() - tick)
                self._stop.wait(max(0.0, drift))


# Standalone CLI
def _cli():
    ap = argparse.ArgumentParser(
        description="Sample NVIDIA GPU usage to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", default="gpu_usage.csv", help="Output CSV path")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Seconds between samples")
    ap.add_argument("--pid", type=int, action="append", default=[],
                    help="PID to track per-process usage for (repeatable). "
                         "Sampling also stops when all tracked PIDs exit "
                         "unless --duration/--forever overrides.")
    ap.add_argument("--gpu", type=int, action="append", default=[],
                    help="Restrict to GPU index (repeatable); default all")
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after this many seconds")
    ap.add_argument("--forever", action="store_true",
                    help="Keep running even if tracked PIDs exit")
    args = ap.parse_args()

    mon = GpuMonitor(
        out_path=args.out,
        interval=args.interval,
        track_pids=args.pid or None,
        gpu_indices=args.gpu or None,
    )

    stop_flag = {"stop": False}

    def handle_sig(signum, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    mon.start()
    print(f"[gpu] sampling every {args.interval}s -> {args.out} "
          f"(Ctrl-C to stop)", file=sys.stderr)
    t0 = time.time()
    try:
        while not stop_flag["stop"]:
            if args.duration is not None and (time.time() - t0) >= args.duration:
                break
            if args.pid and not args.forever and _HAVE_PSUTIL:
                alive = any(psutil.pid_exists(p) for p in args.pid)
                if not alive:
                    print("[gpu] tracked PID(s) exited; stopping",
                          file=sys.stderr)
                    break
            time.sleep(0.2)
    finally:
        mon.stop()


if __name__ == "__main__":
    _cli()