"""run_config(jobs > 1) must write the same results as the sequential run (only timings differ).

Run with `python tests/test_parallel_runner.py` or `pytest tests/`.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_relevance import run_config

TIMING = {"proc_time_s", "log_time_s"}


def test_parallel_runs_match_sequential():
    cfg = json.load(open(ROOT / "configs" / "protocol" / "s1_gauss_abrupt.json"))
    cfg["stream"].update(n=800, drift_at=400)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        s_seq = run_config(cfg, seeds=[0, 1], outdir=a, quiet=True, jobs=1)
        s_par = run_config(cfg, seeds=[0, 1], outdir=b, quiet=True, jobs=4)
        for v in cfg["variants"]:
            for k in s_seq[v]:
                if k not in TIMING:
                    assert json.dumps(s_seq[v][k]) == json.dumps(s_par[v][k]), (v, k)
            for s in (0, 1):
                da = Path(a) / cfg["name"] / v / f"seed_{s}"
                db = Path(b) / cfg["name"] / v / f"seed_{s}"
                ma, mb = json.load(open(da / "metrics.json")), json.load(open(db / "metrics.json"))
                strip = lambda m: json.dumps({k: m[k] for k in m if k not in TIMING})   # NaN-safe compare
                assert strip(ma) == strip(mb), (v, s)
                with np.load(da / "checkpoints.npz") as za, np.load(db / "checkpoints.npz") as zb:
                    for key in za.files:
                        if key != "proc_time_s":
                            assert np.array_equal(za[key], zb[key], equal_nan=True), (v, s, key)
                assert (da / "per_rule.csv").read_text() == (db / "per_rule.csv").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
