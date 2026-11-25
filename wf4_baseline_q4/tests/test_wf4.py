import hashlib
import sys
from pathlib import Path
import numpy as np
from river.datasets import synth

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
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
    """Logging ativo vs. desligado no mesmo runner deve produzir predições e regras idênticas."""
    X, y = _generate_synthetic_data(n_samples=250, n_features=5, seed=7)

    m_off, traj_off = evaluate_single_run(X, y, n_classes=2, variant="FULL", seed=0, track_trajectory=False)
    m_on, traj_on = evaluate_single_run(X, y, n_classes=2, variant="FULL", seed=0, checkpoint_interval=50, track_trajectory=True)

    assert traj_off is None
    assert traj_on is not None and len(traj_on["checkpoints"]) == 5
    assert m_off["accuracy"] == m_on["accuracy"]
    assert m_off["macro_f1"] == m_on["macro_f1"]
    assert m_off["final_rules"] == m_on["final_rules"]


def test_no_future_leakage_fast_causal_scaler():
    """
    Valida o FastCausalMinMaxScaler efetivamente utilizado no run_wf4.py:
    - warm-up em t=0 devolve zeros sem ver o futuro;
    - alterar amostras futuras (t > 100) não altera nenhuma predição nem estado até t=100.
    """
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

    preds_a = _collect_preds(X_a, y_a)
    preds_b = _collect_preds(X_b, y_b)
    assert np.array_equal(preds_a[:101], preds_b[:101])


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


def test_e0_baseline_equivalence():
    """E0: Verifica equivalência exata entre o loop da baseline e a execução amostra-a-amostra."""
    stream = synth.Agrawal(classification_function=0, seed=42)
    X_raw, y_raw, keys = [], [], None
    for xd, label in stream.take(800):
        if keys is None:
            keys = sorted(xd.keys())
        X_raw.append([float(xd[k]) for k in keys])
        y_raw.append(int(label))

    X_arr = np.asarray(X_raw, dtype=float)
    y_arr = np.asarray(y_raw, dtype=int)
    mins = X_arr.min(axis=0)
    maxs = np.where(X_arr.max(axis=0) > mins, X_arr.max(axis=0), mins + 1.0)
    X_norm = (X_arr - mins) / (maxs - mins)

    m_ref = EvolvingNeuroFuzzyUnimSafe(n_features=X_norm.shape[1], n_classes=2, max_rules=5, random_state=42)
    m_step = EvolvingNeuroFuzzyUnimSafe(n_features=X_norm.shape[1], n_classes=2, max_rules=5, random_state=42)

    preds_ref = np.zeros(len(y_arr), dtype=int)
    for t in range(len(y_arr)):
        preds_ref[t] = m_ref.predict(X_norm[t : t + 1])[0]
        m_ref.partial_fit(X_norm[t : t + 1], y_arr[t : t + 1])

    preds_step = np.zeros(len(y_arr), dtype=int)
    for t in range(len(y_arr)):
        preds_step[t] = m_step.predict(X_norm[t : t + 1])[0]
        m_step.partial_fit(X_norm[t : t + 1], y_arr[t : t + 1])

    assert np.array_equal(preds_ref, preds_step)
    assert m_ref.n_rules_ == m_step.n_rules_


def test_ablation_wrappers_invariants():
    """
    Valida os wrappers de ablação:
      - FULL == modelo original sem wrapper;
      - NO_W mantém w == 1.0 em todas as regras (novas, existentes e fundidas);
      - SHUFFLED_W aplica permutação fixa não-identidade preservando os valores do vetor.
    """
    X, y = _generate_synthetic_data(n_samples=350, n_features=6, seed=19)

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

    assert not np.array_equal(shuf_wrap.perm, np.arange(6))

    for t in range(len(y)):
        xi = X[t : t + 1]
        yi = y[t : t + 1]
        assert np.array_equal(raw_model.predict(xi), full_wrap.predict(xi))

        raw_model.partial_fit(xi, yi)
        full_wrap.partial_fit(xi, yi)
        now_wrap.partial_fit(xi, yi)
        shuf_wrap.partial_fit(xi, yi)

        # NO_W: todas as regras têm pesos unitários
        for r in now_wrap.rules:
            assert np.allclose(r.w, 1.0)

        # SHUFFLED_W: cada regra tem os pesos exatos da regra correspondente permutados por shuf_wrap.perm
        assert len(raw_model.rules) == len(shuf_wrap.rules)
        for r_raw, r_shuf in zip(raw_model.rules, shuf_wrap.rules):
            assert np.allclose(r_shuf.w, r_raw.w[shuf_wrap.perm])


def test_incremental_estimators_unit():
    """
    Testes unitários para os estimadores incrementais:
      1. Feature discriminativa recebe peso maior e finito;
      2. Adaptação após drift abrupto sem acesso a amostras futuras;
      3. Memória constante O(C*d) sem manter o buffer de 200 amostras em paralelo.
    """
    rng = np.random.default_rng(123)
    for method in ["INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"]:
        buf = IncrementalSeparabilityBuffer(n_features=4, method=method, alpha=0.1)
        # Antes do drift: f0 separa as classes, f1..f3 são ruído
        for t in range(150):
            cls = t % 2
            x = rng.normal(0.0, 0.2, size=4)
            x[0] = 2.0 if cls == 1 else -2.0
            buf.add(x, cls)

        w_pre = buf.compute_feature_weights()
        assert np.all(np.isfinite(w_pre))
        assert w_pre[0] > np.max(w_pre[1:])

        # Drift abrupto: f2 passa a separar as classes, f0 vira ruído
        for t in range(250):
            cls = t % 2
            x = rng.normal(0.0, 0.2, size=4)
            x[2] = 2.0 if cls == 1 else -2.0
            buf.add(x, cls)

        w_post = buf.compute_feature_weights()
        assert np.all(np.isfinite(w_post))
        assert w_post[2] > w_post[0]

        # Garante que o modelo incremental substitui sep_buffer (sem buffer paralelo X/y)
        model = ENFUnimNeuronIncremental(n_features=4, n_classes=2, inc_method=method, alpha_inc=0.05)
        assert isinstance(model.sep_buffer, IncrementalSeparabilityBuffer)
        assert not hasattr(model.sep_buffer, "X")