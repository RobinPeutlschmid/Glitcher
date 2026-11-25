#!/usr/bin/env python3
"""Memory over a long render, one process, frame by frame: finds leaks (memory that keeps
growing after the caches are full) feature by feature.

    python tools/memcheck.py FRAMES -n 80 -- [glitch.py options...]

Prints a JSON line: frames, rss_mb per frame, peak_mb, and the growth (MB per 10 frames) over
the second half of the run, when every cache is already full. A leak-free render is ~flat."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import glitch  # noqa: E402


def rss_mb():
    """Resident memory of this process in MB (Linux, Windows, macOS)."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                 ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2 ** 20
    import resource                                             # macOS: peak only, in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def main():
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(usage="memcheck.py FRAMES [-n N] -- [glitch.py options]")
    ap.add_argument("input")
    ap.add_argument("-n", type=int, default=80, help="frames to render (default 80)")
    a = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    out = Path(os.environ.get("MEMCHECK_OUT", "/tmp/memcheck_out"))
    args = glitch.build_parser().parse_args([a.input, "-o", str(out), "-s", "3", *rest])
    P = glitch.plan_render(args)
    work = P["work"][:a.n]
    trace, t0 = [], time.time()
    for job in work:
        glitch._render(job)
        trace.append(round(rss_mb(), 1))
    half = trace[len(trace) // 2:]
    k = len(half)
    xs = range(k)
    mx, my = (k - 1) / 2, sum(half) / k
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, half)) / max(1e-9, sum((x - mx) ** 2 for x in xs))
    print(json.dumps({"frames": len(work), "peak_mb": max(trace), "end_mb": trace[-1],
                      "growth_mb_per_10_frames": round(slope * 10, 2),
                      "sec_per_frame": round((time.time() - t0) / len(work), 3), "rss_mb": trace}))


if __name__ == "__main__":
    main()
