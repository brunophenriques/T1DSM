import hashlib
import sys
from pathlib import Path
import numpy as np
from river import preprocessing
from river.datasets import synth

# Garante que a raiz do repositório está no sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


class OnlineMinMaxScaler:
    """Wrapper estritamente causal (sem leakage futuro) sobre river.preprocessing.MinMaxScaler."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.scaler = preprocessing.MinMaxScaler()

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        x_dict = {j: float(x[j]) for j in range(self.n_features)}
        xt_dict = self.scaler.transform_one(x_dict)
        return np.array([xt_dict.get(j, 0.0) for j in range(self.n_features)], dtype=float)

    def learn_one(self, x: np.ndarray) -> None:
        x_dict = {j: float(x[j]) for j in range(self.n_features)}
        self.scaler.learn_one(x_dict)


def _generate_synthetic_data(n_samples: int = 300, n_features: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = ((X[:, 0] + 0.8 * X[:, 1]) > 0.0).astype(int)
    return X, y


def run_prequential_stream(X: np.ndarray, y: np.ndarray, model, use_online_scaler: bool = True, log_every: int = 0):
    """
    Executa avaliação estritamente test-then-train.
    Retorna predições e (opcionalmente) logs não-intrusivos de pesos e regras.
    """
    n_samples, n_features = X.shape
    scaler = OnlineMinMaxScaler(n_features) if use_online_scaler else None
    preds = np.zeros(n_samples, dtype=int)

    logs = []
    prev_rule_ids = set()
    created_cum = 0
    deleted_cum = 0

    for t in range(n_samples):
        xt = X[t]
        yt = int(y[t])

        # 1. Transformar usando APENAS o passado (x_0 ... x_{t-1})
        xt_scaled = scaler.transform_one(xt) if scaler is not None else xt

        # 2. Prever ANTES de aprender o rótulo y_t
        preds[t] = int(model.predict(xt_scaled.reshape(1, -1))[0])

        # 3. Atualizar scaler e modelo com (x_t, y_t)
        if scaler is not None:
            scaler.learn_one(xt)
        model.partial_fit(xt_scaled.reshape(1, -1), np.array([yt], dtype=int))

        # 4. Logging passivo (apenas leitura de estado)
        if log_every > 0:
            curr_rule_ids = {id(r) for r in model.rules}
            created_cum += len(curr_rule_ids - prev_rule_ids)
            deleted_cum += len(prev_rule_ids - curr_rule_ids)
            prev_rule_ids = curr_rule_ids

            if (t + 1) % log_every == 0:
                logs.append({
                    "t": t + 1,
                    "w": model.current_feature_weights.copy(),
                    "active_rules": model.n_rules_,
                    "created_cum": created_cum,
                    "deleted_cum": deleted_cum,
                })

    return preds, logs


def test_non_intrusive_logging():
    """Logging ativo vs. desligado deve produzir predições e regras 100% idênticas."""
    X, y = _generate_synthetic_data(n_samples=300, n_features=5, seed=7)

    model_off = EvolvingNeuroFuzzyUnimSafe(n_features=5, n_classes=2, random_state=42)
    model_on = EvolvingNeuroFuzzyUnimSafe(n_features=5, n_classes=2, random_state=42)

    preds_off, logs_off = run_prequential_stream(X, y, model_off, log_every=0)
    preds_on, logs_on = run_prequential_stream(X, y, model_on, log_every=50)

    assert len(logs_off) == 0
    assert len(logs_on) == 6
    assert np.array_equal(preds_off, preds_on)
    assert model_off.n_rules_ == model_on.n_rules_
    assert np.allclose(model_off.current_feature_weights, model_on.current_feature_weights)


def test_no_future_leakage():
    """Corromper amostras futuras (t > 100) não pode alterar nenhuma predição até t = 100."""
    X_a, y_a = _generate_synthetic_data(n_samples=220, n_features=4, seed=11)
    X_b = X_a.copy()
    y_b = y_a.copy()
    X_b[101:] = X_b[101:] * 500.0 + 99.0
    y_b[101:] = 1 - y_b[101:]

    m_a = EvolvingNeuroFuzzyUnimSafe(n_features=4, n_classes=2, random_state=0)
    m_b = EvolvingNeuroFuzzyUnimSafe(n_features=4, n_classes=2, random_state=0)

    preds_a, _ = run_prequential_stream(X_a, y_a, m_a, use_online_scaler=True)
    preds_b, _ = run_prequential_stream(X_b, y_b, m_b, use_online_scaler=True)

    assert np.array_equal(preds_a[:101], preds_b[:101])


def test_determinism_csv(tmp_path):
    """Mesma configuração e seed devem produzir exatamente o mesmo ficheiro CSV."""
    X, y = _generate_synthetic_data(n_samples=250, n_features=5, seed=3)

    def _run_to_csv(file_path: Path) -> str:
        model = EvolvingNeuroFuzzyUnimSafe(n_features=5, n_classes=2, random_state=42)
        preds, _ = run_prequential_stream(X, y, model, use_online_scaler=True)
        lines = ["t,y_true,y_pred"] + [f"{i},{y[i]},{preds[i]}" for i in range(len(y))]
        file_path.write_text("\n".join(lines), encoding="utf-8")
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    h1 = _run_to_csv(tmp_path / "run_a.csv")
    h2 = _run_to_csv(tmp_path / "run_b.csv")
    assert h1 == h2


def test_e0_baseline_equivalence():
    """
    E0: Verifica que o loop amostra-a-amostra com pré-processamento idêntico
    reproduz exatamente a acurácia e o número de regras do protocolo original.
    """
    stream = synth.Agrawal(classification_function=0, seed=42)
    X_raw, y_raw = [], []
    for x_dict, label in stream.take(1000):
        keys = sorted(x_dict.keys())
        X_raw.append([float(x_dict[k]) for k in keys])
        y_raw.append(int(label))

    X_arr = np.asarray(X_raw, dtype=float)
    y_arr = np.asarray(y_raw, dtype=int)

    # Normalização fixa pré-calculada apenas para testar equivalência E0 do motor do modelo
    mins = X_arr.min(axis=0)
    maxs = np.where(X_arr.max(axis=0) > mins, X_arr.max(axis=0), mins + 1.0)
    X_norm = (X_arr - mins) / (maxs - mins)

    m_orig_style = EvolvingNeuroFuzzyUnimSafe(n_features=X_norm.shape[1], n_classes=2, max_rules=5, random_state=42)
    m_stream_style = EvolvingNeuroFuzzyUnimSafe(n_features=X_norm.shape[1], n_classes=2, max_rules=5, random_state=42)

    preds_orig = np.zeros(len(y_arr), dtype=int)
    for t in range(len(y_arr)):
        preds_orig[t] = m_orig_style.predict(X_norm[t : t + 1])[0]
        m_orig_style.partial_fit(X_norm[t : t + 1], y_arr[t : t + 1])

    preds_stream, _ = run_prequential_stream(X_norm, y_arr, m_stream_style, use_online_scaler=False)

    assert np.array_equal(preds_orig, preds_stream)
    assert m_orig_style.n_rules_ == m_stream_style.n_rules_