"""Run once before the Elec2 experiments:  python wf1_stability/check_elec2_ranges.py
Shows whether Elec2 is already in [0,1] (then scaling='none' is leak-free and distortion-free), whether any
feature is monotone (expanding min-max would turn it into a constant), and the trivial-baseline accuracies."""
import numpy as np
from river import datasets

N = 10000
rows, labels, names = [], [], None
for i, (x, y) in enumerate(datasets.Elec2()):
    if i >= N:
        break
    names = names or list(x.keys()); rows.append([x[f] for f in names]); labels.append(y)
X = np.asarray(rows, float); y = np.array([list(dict.fromkeys(labels)).index(v) for v in labels])
print(f"{'feature':14s} {'min':>9s} {'max':>9s} monotone")
for j, n in enumerate(names):
    d = np.diff(X[:, j]); mono = np.ptp(X[:, j]) > 0 and (np.all(d >= 0) or np.all(d <= 0))
    print(f"{n:14s} {X[:, j].min():9.4f} {X[:, j].max():9.4f} {'YES' if mono else ''}")
print("all features inside [0,1]:", bool(X.min() >= 0 and X.max() <= 1))
_, c = np.unique(y, return_counts=True)
print(f"majority-class rate {c.max() / len(y):.3f} | repeat-last-label rate {np.mean(y[1:] == y[:-1]):.3f}")
