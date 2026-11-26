from collections import Counter
import numpy as np


def attach_rule_event_counter(model):
    """
    Instrumenta _create_rule e _maybe_merge_last_rule na instância sem editar
    ENF_UnimNeuron_Safe.py, contando criações, podas imediatas e fusões reais.
    """
    if getattr(model, "_has_rule_counter", False):
        return model
    model.rules_created_exact = 0
    model.rules_evicted_exact = 0
    model.rules_merged_exact = 0

    orig_create = model._create_rule
    orig_merge = model._maybe_merge_last_rule

    def counted_create(x, y, sigma):
        n_before = len(model.rules)
        orig_create(x, y, sigma)
        n_after = len(model.rules)
        model.rules_created_exact += 1
        if n_after <= n_before:
            model.rules_evicted_exact += 1

    def counted_merge():
        n_before = len(model.rules)
        orig_merge()
        n_after = len(model.rules)
        if n_after < n_before:
            model.rules_merged_exact += (n_before - n_after)

    model._create_rule = counted_create
    model._maybe_merge_last_rule = counted_merge
    model._has_rule_counter = True
    return model


class MajorityClassBaseline:
    """Baseline prequencial que prevê sempre a classe mais frequente observada até t-1."""

    def __init__(self, n_classes: int = 2):
        self.n_classes = int(n_classes)
        self.counts = Counter()

    @property
    def n_rules_(self) -> int:
        return 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred_cls = self.counts.most_common(1)[0][0] if self.counts else 0
        return np.full(X.shape[0], int(pred_cls), dtype=int)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for yi in np.asarray(y, dtype=int):
            self.counts[int(yi)] += 1


class NoChangeBaseline:
    """Baseline prequencial de persistência: prevê y_hat(t) = y(t-1)."""

    def __init__(self, default_class: int = 0):
        self.last_y = int(default_class)

    @property
    def n_rules_(self) -> int:
        return 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[0], self.last_y, dtype=int)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=int)
        if y_arr.size > 0:
            self.last_y = int(y_arr[-1])


class AblationWeightWrapper:
    """
    Wrapper não-intrusivo sobre EvolvingNeuroFuzzyUnimSafe (nunca edita o core):
      - FULL: mantém o modelo original intacto (previsões e regras idênticas).
      - NO_W: força pesos unitários (1.0) em regras novas, existentes e fundidas.
      - SHUFFLED_W: aplica um derangement fixo por execução (pi(j) != j para todo j,
        determinado pela seed, nunca escolhido pela accuracy), preservando os valores
        do vetor de pesos na criação, atualização e fusão de regras.
    """

    def __init__(self, base_model, mode: str = "FULL", seed: int = 0):
        assert mode in {"FULL", "NO_W", "SHUFFLED_W"}, f"Modo inválido: {mode}"
        self.model = base_model
        self.mode = mode
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        d = self.model.n_features
        perm = self.rng.permutation(d)
        if d > 1:
            identity = np.arange(d)
            while np.any(perm == identity):
                perm = self.rng.permutation(d)
        self.perm = perm
        self._ones = np.ones(d, dtype=float)

        self._apply_to_global_and_existing()

    @property
    def n_rules_(self) -> int:
        return self.model.n_rules_

    @property
    def rules(self):
        return self.model.rules

    @property
    def current_feature_weights(self) -> np.ndarray:
        return self.model.current_feature_weights

    @property
    def rules_created_exact(self) -> int:
        return int(getattr(self.model, "rules_created_exact", 0))

    @property
    def rules_evicted_exact(self) -> int:
        return int(getattr(self.model, "rules_evicted_exact", 0))

    @property
    def rules_merged_exact(self) -> int:
        return int(getattr(self.model, "rules_merged_exact", 0))

    def _transform_vec(self, w: np.ndarray) -> np.ndarray:
        if self.mode == "NO_W":
            return self._ones.copy()
        if self.mode == "SHUFFLED_W":
            return np.asarray(w, dtype=float)[self.perm].copy()
        return np.asarray(w, dtype=float).copy()

    def _apply_to_global_and_existing(self) -> None:
        if self.mode == "FULL":
            return
        self.model.current_feature_weights = self._transform_vec(self.model.current_feature_weights)
        for r in self.model.rules:
            r.w = self._transform_vec(r.w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mode == "FULL":
            self.model.partial_fit(X, y)
            return

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        for xi, yi in zip(X, y):
            self.model.partial_fit(xi.reshape(1, -1), np.array([int(yi)], dtype=int))
            if self.model.K % 50 == 0:
                self.model.current_feature_weights = self._transform_vec(
                    self.model.current_feature_weights
                )