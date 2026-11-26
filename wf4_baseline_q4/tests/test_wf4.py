import hashlib
import sys
from pathlib import Path
import numpy as np
from river.datasets import synth

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe, ENFRuleSafe
from exp_sota_comparison import min_max_scale, prequential_nf_generic
from wf4_baseline_q4.run_wf4 import FastCausalMinMaxScaler, evaluate_single_run
from wf4_baseline_q4.wrappers import AblationWeightWrapper
from wf4_baseline_q4.incremental_weights import (
    ENFUnimNeuronIncremental,
    IncrementalSeparabilityBuffer,
)


def _generate_synthetic_data(n_samples: int = 300, n_features: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = ((X[:, 0] + 0.8 * X[:, 1]) > 0.0).astype(int)
    return X, y


def test_non_intrusive_logging():
    """Logging ativo vs. desligado no mesmo runner deve produzir predições, regras e estado idênticos."""
    X, y = _generate_synthetic_data(n_samples=250, n_features=5, seed=7)

    m_off, traj_off = evaluate_single_run(X, y, n_classes=2, variant="FULL", seed=0, track_trajectory=False)
    m_on, traj_on = evaluate_single_run(
        X, y, n_classes=2, variant="FULL", seed=0, checkpoint_interval=50, track_trajectory=True
    )

    assert traj_off is None
    assert traj_on is not None and len(traj_on["checkpoints"]) == 5
    assert m_off["accuracy"] == m_on["accuracy"]
    assert m_off["macro_f1"] == m_on["macro_f1"]
    assert m_off["final_rules"] == m_on["final_rules"]


def test_no_future_leakage_fast_causal_scaler():
    """Valida warm-up causal em t=0 e ausência de leakage de amostras futuras (t > 100)."""
    X_a, y_a = _generate_synthetic_data(n_samples=220, n_features=4, seed=11)
    X_b = X_a.copy()
    y_b = y_a.copy()
    X_b[101:] = X_b[101:] * 500.0 + 99.0
    y_b[101:] = 1 - y_b[101:]

    def _collect_preds(X_in, y_in):
        scaler = FastCausalMinMaxScaler(X_in.shape[1])
        model = EvolvingNeuroFuzzyUnimSafe(n_features=X_in.shape[1], n_classes=2, random_state=42)
        preds = np.zeros(len(y_in), dtype=int)
        for t in range(len(y_in)):
            xs = scaler.transform_one(X_in[t])
            if t == 0:
                assert np.allclose(xs, 0.0)
            preds[t] = int(model.predict(xs.reshape(1, -1))[0])
            scaler.learn_one(X_in[t])
            model.partial_fit(xs.reshape(1, -1), np.array([int(y_in[t])], dtype=int))
        return preds

    assert np.array_equal(_collect_preds(X_a, y_a)[:101], _collect_preds(X_b, y_b)[:101])


def test_determinism_csv(tmp_path):
    """Repetir configuração e seed gera CSV determinístico idêntico."""
    X, y = _generate_synthetic_data(n_samples=250, n_features=5, seed=3)

    def _run_to_csv(path: Path) -> str:
        m, _ = evaluate_single_run(X, y, n_classes=2, variant="FULL", seed=3, track_trajectory=False)
        det_keys = ["accuracy", "macro_f1", "final_rules", "rules_created_id_approx", "rules_deleted_id_approx"]
        content = ",".join(det_keys) + "\n" + ",".join(str(m[k]) for k in det_keys)
        path.write_text(content, encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert _run_to_csv(tmp_path / "r1.csv") == _run_to_csv(tmp_path / "r2.csv")


def test_e0_real_baseline_vs_causal():
    """
    E0 real: compara exp_sota_comparison.py (prequential_nf_generic + min_max_scale offline)
    diretamente contra evaluate_single_run() utilizada no benchmark, quantificando depois
    o efeito de ativar o FastCausalMinMaxScaler.
    """
    stream = synth.Agrawal(classification_function=0, seed=42)
    X_raw, y_raw, keys = [], [], None
    for xd, label in stream.take(1000):
        if keys is None:
            keys = sorted(xd.keys())
        X_raw.append([float(xd[k]) for k in keys])
        y_raw.append(int(label))
    X_arr = np.asarray(X_raw, dtype=float)
    y_arr = np.asarray(y_raw, dtype=int)

    # 1. Execução oficial via exp_sota_comparison.py (offline min_max_scale, max_rules=5)
    X_offline, _, _ = min_max_scale(X_arr)
    m_orig = EvolvingNeuroFuzzyUnimSafe(n_features=X_arr.shape[1], n_classes=2, max_rules=5, random_state=42)
    res_orig = prequential_nf_generic(m_orig, X_offline, y_arr, desc="E0_orig")

    # 2. evaluate_single_run() real em condições idênticas (offline min_max_scale, y_hat=0 em t=0)
    m_equiv, _ = evaluate_single_run(
        X_offline,
        y_arr,
        n_classes=2,
        variant="FULL_REF",
        seed=0,
        use_causal_scaler=False,
        first_step_zero_pred=True,
    )

    assert round(float(res_orig["accuracy"][-1]), 6) == m_equiv["accuracy"]
    assert res_orig["n_rules_final"] == m_equiv["final_rules"]

    # 3. evaluate_single_run() com FastCausalMinMaxScaler (zero leakage)
    m_causal, _ = evaluate_single_run(
        X_arr,
        y_arr,
        n_classes=2,
        variant="FULL_REF",
        seed=0,
        use_causal_scaler=True,
        first_step_zero_pred=False,
    )
    assert 0.0 <= m_causal["accuracy"] <= 1.0
    assert m_causal["final_rules"] <= 5


def test_ablation_wrappers_invariants():
    """
    Valida invariantes das ablações sem pressupor igualdade no número de regras entre
    FULL e SHUFFLED_W (já que permutar w altera a ativação e pode alterar fusões/criações):
      - FULL == modelo original sem wrapper;
      - NO_W mantém w == 1.0 em todas as regras novas, existentes e após fusão;
      - SHUFFLED_W aplica permutação fixa não-identidade preservando o multiconjunto de pesos
        na criação e após fusão controlada de regras.
    """
    X, y = _generate_synthetic_data(n_samples=300, n_features=6, seed=19)

    raw_model = EvolvingNeuroFuzzyUnimSafe(n_features=6, n_classes=2, max_rules=6, random_state=42)
    full_wrap = AblationWeightWrapper(
        EvolvingNeuroFuzzyUnimSafe(n_features=6, n_classes=2, max_rules=6, random_state=42),
        mode="FULL",
        seed=5,
    )
    now_wrap = AblationWeightWrapper(
        EvolvingNeuroFuzzyUnimSafe(n_features=6, n_classes=2, max_rules=6, random_state=42),
        mode="NO_W",
        seed=5,
    )
    shuf_wrap = AblationWeightWrapper(
        EvolvingNeuroFuzzyUnimSafe(n_features=6, n_classes=2, max_rules=6, random_state=42),
        mode="SHUFFLED_W",
        seed=5,
    )

    assert np.all(shuf_wrap.perm != np.arange(6))

    for t in range(len(y)):
        xi = X[t : t + 1]
        yi = y[t : t + 1]
        assert np.array_equal(raw_model.predict(xi), full_wrap.predict(xi))

        raw_model.partial_fit(xi, yi)
        full_wrap.partial_fit(xi, yi)
        now_wrap.partial_fit(xi, yi)
        shuf_wrap.partial_fit(xi, yi)

        assert len(raw_model.rules) == len(full_wrap.rules)
        for r in now_wrap.rules:
            assert np.allclose(r.w, 1.0)
        for r in shuf_wrap.rules:
            assert np.all(np.isfinite(r.w))
            assert r.w.shape == (6,)

    # Cenário controlado de fusão (_maybe_merge_last_rule) para SHUFFLED_W e NO_W
    w_base_1 = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.3])
    w_base_2 = np.array([0.9, 0.3, 0.5, 0.7, 0.2, 0.8])
    perm = shuf_wrap.perm

    shuf_merge_model = EvolvingNeuroFuzzyUnimSafe(n_features=6, n_classes=2, min_rules_for_merge=1, tau_merge=0.5)
    shuf_merge_wrap = AblationWeightWrapper(shuf_merge_model, mode="SHUFFLED_W", seed=5)
    shuf_merge_model.rules = [
        ENFRuleSafe(center=np.zeros(6), sigma=1.0, support=12, w=w_base_1[perm].copy(), class_counts=np.array([6.0, 6.0])),
        ENFRuleSafe(center=np.zeros(6), sigma=1.0, support=12, w=w_base_2[perm].copy(), class_counts=np.array([6.0, 6.0])),
    ]
    shuf_merge_model._maybe_merge_last_rule()
    assert len(shuf_merge_model.rules) == 1
    expected_merged = 0.5 * w_base_1[perm] + 0.5 * w_base_2[perm]
    assert np.allclose(shuf_merge_model.rules[0].w, expected_merged)
    assert np.allclose(np.sort(shuf_merge_model.rules[0].w), np.sort(0.5 * w_base_1 + 0.5 * w_base_2))

    now_merge_model = EvolvingNeuroFuzzyUnimSafe(n_features=6, n_classes=2, min_rules_for_merge=1, tau_merge=0.5)
    now_merge_wrap = AblationWeightWrapper(now_merge_model, mode="NO_W", seed=5)
    now_merge_model.rules = [
        ENFRuleSafe(center=np.zeros(6), sigma=1.0, support=12, w=np.ones(6), class_counts=np.array([6.0, 6.0])),
        ENFRuleSafe(center=np.zeros(6), sigma=1.0, support=12, w=np.ones(6), class_counts=np.array([6.0, 6.0])),
    ]
    now_merge_model._maybe_merge_last_rule()
    assert len(now_merge_model.rules) == 1
    assert np.allclose(now_merge_model.rules[0].w, 1.0)


def test_incremental_estimators_unit():
    """Valida separação conhecida, adaptação a drift sem leakage e memória O(C*d) sem buffer paralelo."""
    rng = np.random.default_rng(123)
    for method in ["INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"]:
        buf = IncrementalSeparabilityBuffer(n_features=4, method=method, alpha=0.1)
        for t in range(150):
            cls = t % 2
            x = rng.normal(0.0, 0.2, size=4)
            x[0] = 2.0 if cls == 1 else -2.0
            buf.add(x, cls)

        w_pre = buf.compute_feature_weights()
        assert np.all(np.isfinite(w_pre))
        assert w_pre[0] > np.max(w_pre[1:])

        for t in range(250):
            cls = t % 2
            x = rng.normal(0.0, 0.2, size=4)
            x[2] = 2.0 if cls == 1 else -2.0
            buf.add(x, cls)

        w_post = buf.compute_feature_weights()
        assert np.all(np.isfinite(w_post))
        assert w_post[2] > w_post[0]

        model = ENFUnimNeuronIncremental(n_features=4, n_classes=2, inc_method=method, alpha_inc=0.05)
        assert isinstance(model.sep_buffer, IncrementalSeparabilityBuffer)
        assert not hasattr(model.sep_buffer, "X")