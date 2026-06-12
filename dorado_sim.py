#!/usr/bin/env python3
"""
Simulate streaming Dorado basecalling with pore-driven timing, while supporting ability to
record NVIDIA GPU resource usage during the basecalling step.

Timing model:
    completion_time = (start_sample + sample_count) / sample_rate
Reads are emitted in temporal order; each interval releases up to X reads
whose sequencing had completed by the current simulated time.

GPU monitoring:
    The dorado basecaller subprocess is launched via Popen so its PID is
    known, and a background GpuMonitor (see gpu_monitor.py) samples GPU and
    per-process usage to a CSV for the duration of the basecall.

Usage:
    python dorado_sim.py /path/to/pod5_dir \
        --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
        --reads-per-batch 50 --interval 10 --speed 60 \
        --outdir ./sim_output \
        --gpu-csv ./sim_output/gpu_usage.csv --gpu-interval 1.0

Dependencies:
    - dorado on PATH (or pass --dorado /path/to/dorado)
    - pod5            (pip install pod5)
    - nvidia-ml-py    (pip install nvidia-ml-py)   [for GPU monitoring]
    - psutil          (pip install psutil)         [optional, tracks children]
    - Python 3.8+
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pod5 as p5

# GPU monitor is optional: the sim runs without it.
try:
    from gpu_monitor import GpuMonitor
    _HAVE_GPU_MONITOR = True
except ImportError:
    _HAVE_GPU_MONITOR = False


def find_dorado(explicit=None):
    """User specified dorado installation path. PATH installation is default"""
    if explicit:
        if not shutil.which(explicit) and not Path(explicit).is_file():
            sys.exit(f"ERROR: dorado not found at: {explicit}")
        return explicit
    found = shutil.which("dorado")
    if not found:
        sys.exit("ERROR: 'dorado' not on PATH. Install it or pass --dorado.")
    return found


def run_dorado_basecall(dorado, model, pod5_dir, work_dir, extra_args=None,
                        gpu_csv=None, gpu_interval=1.0, gpu_indices=None):
    """
    Run dorado basecaller over the pod5 directory -> single FASTQ.

    If gpu_csv is set and gpu_monitor is importable, GPU usage is sampled to
    that CSV for the duration of the basecall, with the dorado PID tracked for
    per-process metrics.
    """
    fastq_path = work_dir / "calls.fastq"
    cmd = [dorado, "basecaller", model, str(pod5_dir), "--emit-fastq"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[dorado] running: {' '.join(cmd)}", file=sys.stderr)

    monitor = None
    with open(fastq_path, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.PIPE,
                                text=True)

        # Start GPU monitoring now that we have the dorado PID.
        if gpu_csv:
            if not _HAVE_GPU_MONITOR:
                print("[gpu] WARNING: gpu_monitor.py not importable; "
                      "skipping GPU monitoring.", file=sys.stderr)
            else:
                try:
                    monitor = GpuMonitor(
                        out_path=gpu_csv,
                        interval=gpu_interval,
                        track_pids=[proc.pid],
                        gpu_indices=gpu_indices,
                    ).start()
                    print(f"[gpu] monitoring dorado PID {proc.pid} -> {gpu_csv}",
                          file=sys.stderr)
                except Exception as e:  # NVML init failure, no GPU, etc.
                    print(f"[gpu] WARNING: could not start GPU monitor: {e}",
                          file=sys.stderr)
                    monitor = None

        try:
            _, stderr = proc.communicate()
        finally:
            if monitor is not None:
                monitor.stop()

    if proc.returncode != 0:
        if stderr:
            sys.stderr.write(stderr)
        sys.exit(f"ERROR: dorado exited with code {proc.returncode}")
    if fastq_path.stat().st_size == 0:
        sys.exit("ERROR: dorado produced an empty FASTQ.")
    return fastq_path


def build_timing_index(pod5_dir):
    """
    Map read_id -> sequencing completion time (seconds into the run).
    completion = (start_sample + sample_count) / sample_rate
    """
    pod5_dir = Path(pod5_dir)
    if not pod5_dir.is_dir():
        sys.exit(f"ERROR: input directory does not exist: {pod5_dir}")
    files = sorted(pod5_dir.rglob("*.pod5"))
    if not files:
        sys.exit(f"ERROR: no .pod5 files found under {pod5_dir}")

    timing = {}
    print(f"[pod5] indexing timing from {len(files)} file(s)...", file=sys.stderr)
    with p5.DatasetReader(pod5_dir, recursive=True) as dataset:
        for rec in dataset.reads():
            sample_rate = float(rec.run_info.sample_rate)
            if sample_rate <= 0:
                continue
            start = int(rec.start_sample)
            count = int(rec.sample_count)
            completion = (start + count) / sample_rate
            timing[str(rec.read_id)] = {
                "completion_s": completion,
                "duration_s": count / sample_rate,
                "samples": count,
            }
    if not timing:
        sys.exit("ERROR: no reads found in pod5 files.")
    print(f"[pod5] indexed {len(timing)} reads", file=sys.stderr)
    return timing


def parse_fastq_by_id(fastq_path):
    """Return dict: read_id -> 4-line FASTQ record string."""
    opener = gzip.open if str(fastq_path).endswith(".gz") else open
    records = {}
    with opener(fastq_path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()
            if not qual:
                break
            read_id = header[1:].split(maxsplit=1)[0].strip()
            records[read_id] = header + seq + plus + qual
    return records


def main():
    ap = argparse.ArgumentParser(
        description="Simulate live Dorado basecalling with pod5-driven timing "
                    "and GPU usage recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("pod5_dir", help="Directory containing .pod5 files")
    ap.add_argument("--model", required=True, help="Dorado model name or path")
    ap.add_argument("--reads-per-batch", type=int, default=50,
                    help="X: max reads released per interval")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="N: real seconds between release checks")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Time acceleration. 1.0 = real time.")
    ap.add_argument("--outdir", default="./sim_output",
                    help="Where to write per-batch FASTQ files")
    ap.add_argument("--dorado", default=None, help="Path to dorado executable")
    ap.add_argument("--cache-fastq", default=None,
                    help="Reuse an already-basecalled FASTQ instead of re-running dorado")
    ap.add_argument("--keep-temp", action="store_true",
                    help="Keep the intermediate full FASTQ")
    ap.add_argument("--dorado-arg", action="append", default=[],
                    help="Extra arg passed through to dorado (repeatable)")
    
    # GPU monitoring options.
    ap.add_argument("--gpu-csv", default=None,
                    help="If set, record GPU usage during basecalling to this CSV")
    ap.add_argument("--gpu-interval", type=float, default=1.0,
                    help="Seconds between GPU samples")
    ap.add_argument("--gpu", type=int, action="append", default=[],
                    help="Restrict GPU monitoring to this index (repeatable)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    timing = build_timing_index(args.pod5_dir)

    if args.cache_fastq:
        fastq_path = Path(args.cache_fastq)
        if not fastq_path.is_file():
            sys.exit(f"ERROR: --cache-fastq not found: {fastq_path}")
        work_dir = None
        print(f"[sim] using cached FASTQ: {fastq_path}", file=sys.stderr)
        if args.gpu_csv:
            print("[gpu] NOTE: --cache-fastq skips basecalling, so there is "
                  "no dorado run to monitor.", file=sys.stderr)
    else:
        dorado = find_dorado(args.dorado)
        work_dir = Path(tempfile.mkdtemp(prefix="dorado_sim_"))
        fastq_path = run_dorado_basecall(
            dorado, args.model, Path(args.pod5_dir), work_dir, args.dorado_arg,
            gpu_csv=args.gpu_csv, gpu_interval=args.gpu_interval,
            gpu_indices=(args.gpu or None),
        )

    records = parse_fastq_by_id(fastq_path)
    print(f"[sim] parsed {len(records)} basecalled reads", file=sys.stderr)

    reads = []
    missing_timing = 0
    for read_id, fastq_text in records.items():
        info = timing.get(read_id)
        if info is None:
            missing_timing += 1
            continue
        reads.append((info["completion_s"], read_id, fastq_text, info))
    if missing_timing:
        print(f"[sim] WARNING: {missing_timing} basecalled reads had no pod5 "
              f"timing match (skipped)", file=sys.stderr)
    if not reads:
        sys.exit("ERROR: no reads with both basecalls and timing.")

    reads.sort(key=lambda r: r[0])
    run_end = reads[-1][0]
    print(f"[sim] {len(reads)} reads span a simulated run of "
          f"{run_end:.1f}s ({run_end/3600:.2f}h)", file=sys.stderr)
    print(f"[sim] replaying at --speed {args.speed}x, releasing up to "
          f"{args.reads_per_batch} reads every {args.interval}s real time\n",
          file=sys.stderr)

    manifest = []
    idx = 0
    total = 0
    batch_no = 0
    wall_start = time.time()

    while idx < len(reads):
        elapsed_real = time.time() - wall_start
        sim_time = elapsed_real * args.speed

        batch = []
        while (idx < len(reads)
               and reads[idx][0] <= sim_time
               and len(batch) < args.reads_per_batch):
            batch.append(reads[idx])
            idx += 1

        if batch:
            batch_no += 1
            batch_file = outdir / f"batch_{batch_no:05d}.fastq"
            with open(batch_file, "w") as out:
                out.writelines(r[2] for r in batch)
            total += len(batch)
            manifest.append({
                "batch": batch_no,
                "reads": len(batch),
                "cumulative_reads": total,
                "sim_time_s": round(sim_time, 2),
                "real_time_s": round(elapsed_real, 2),
                "first_completion_s": round(batch[0][0], 2),
                "last_completion_s": round(batch[-1][0], 2),
                "file": str(batch_file),
            })
            print(f"[sim] real={elapsed_real:7.1f}s  sim={sim_time:9.1f}s  "
                  f"batch {batch_no:>5}  +{len(batch):>4}  (total {total})",
                  file=sys.stderr)

        if idx >= len(reads):
            break
        time.sleep(args.interval)

    with open(outdir / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"\n[sim] done: {total} reads in {batch_no} batches over "
          f"{time.time() - wall_start:.1f}s real time", file=sys.stderr)
    print(f"[sim] output in {outdir}/", file=sys.stderr)
    if args.gpu_csv and not args.cache_fastq:
        print(f"[gpu] GPU usage time series in {args.gpu_csv}", file=sys.stderr)

    if work_dir and not args.keep_temp:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif work_dir:
        print(f"[sim] kept intermediate FASTQ in {work_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()