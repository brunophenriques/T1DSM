import copy
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from experiments.dynamic_weights import DynamicUnimNeuron
from run_relevance import run_one
from relevance.analysis import analyze

with open(ROOT / "configs" / "mve.json") as f:
    base_config = json.load(f)

SEEDS = list(range(10))

VARIANTS = [
    ("Baseline-200", EvolvingNeuroFuzzyUnimSafe, 200),
    ("Dynamic-200", DynamicUnimNeuron, 200),
    ("Dynamic-100", DynamicUnimNeuron, 100)
]

results = []

for name, model_cls, window in VARIANTS:

    for seed in SEEDS:

        config = copy.deepcopy(base_config)

        config["model"]["buffer_size_separability"] = window

        result = run_one(
            config,
            seed,
            log=True,
            model_cls=model_cls
        )

        stream = result["stream"]

        metrics = analyze(
            result["logger"].arrays(),
            stream.truth,
            stream.drift_points,
            config["analysis"],
            config["checkpoint_every"]
        )

        row = {
            "variant": name,
            "seed": seed,
            "recovery_G": metrics["recovery/G/drift1"],
            "recovery_A": metrics["recovery/A/drift1"],
            "precision_G": metrics["valid/G/prec_at_k/seg1"],
            "precision_A": metrics["valid/A/prec_at_k/seg1"],
            "accuracy": result["acc_final"],
            "runtime": result["proc_time_s"]
        }

        results.append(row)

        print(
            f"{name} | Seed={seed} | "
            f"Recovery G={row['recovery_G']} | "
            f"Recovery A={row['recovery_A']} | "
            f"Accuracy={row['accuracy']:.4f}"
        )

output = ROOT / "results_q2" / "dynamic_comparison"
output.mkdir(parents=True, exist_ok=True)

with open(output / "results.csv", "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {output}")