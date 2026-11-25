"""Memory: a long render must not keep growing once its caches are full (a leak would eat the
RAM of long shots), and the caches must stay inside their budgets."""
import json
import subprocess
import sys

import pytest

import synth
from conftest import ROOT

PROBE = r'''
import sys, json, tracemalloc
sys.path.insert(0, sys.argv[1])
import glitch
args = glitch.build_parser().parse_args([sys.argv[2], "-o", sys.argv[3], "-s", "3", "-i", "0.9",
                                         *json.loads(sys.argv[4])])
P = glitch.plan_render(args)
tracemalloc.start()
mid = end = 0
for i, job in enumerate(P["work"], 1):
    glitch._render(job)
    if i == len(P["work"]) // 2:
        mid = tracemalloc.get_traced_memory()[0]
end = tracemalloc.get_traced_memory()[0]
print(json.dumps({"mid": mid, "end": end, "prep": glitch._PREP_BYTES[0], "src": glitch.SeqHistory._bytes,
                  "feats": len(glitch._FEATS or {}), "motions": len(glitch._MOTION or {})}))
'''

CASES = {
    "time effects": ["--gain", "echo=3,stutter=2,mosh=3"],
    "everything": ["--gain", "echo=2,mosh=2", "--appear", "5:1", "--vanish", "250:1", "--motion", "0.6",
                   "--glitch-pass", "--follow", "0.3", "--pad", "0.1"],
}


@pytest.fixture(scope="module")
def long_seq(tmp_path_factory):
    d = tmp_path_factory.mktemp("long")
    synth.make_frames(d, 320, move=0.5)
    return d


@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_no_growth_after_the_caches_fill(name, long_seq, tmp_path):
    r = subprocess.run([sys.executable, "-c", PROBE, str(ROOT), str(long_seq), str(tmp_path / "o"),
                        json.dumps(CASES[name])], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    m = json.loads(r.stdout.strip().splitlines()[-1])
    growth = (m["end"] - m["mid"]) / 2 ** 20
    assert growth < 4, f"{name}: memory grew {growth:.1f} MB over the second 160 frames (a leak?) {m}"
    assert m["feats"] <= 64 and m["motions"] <= 96                  # count-bounded caches
