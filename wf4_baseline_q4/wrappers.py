from collections import Counter
import numpy as np


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
    Wrapper não-intrusivo O(1) sobre EvolvingNeuroFuzzyUnimSafe.
    Como ENF_UnimNeuron_Safe atualiza current_feature_weights apenas quando K % 50 == 0
    e copia esse vetor ao criar novas regras, transformar current_feature_weights nesses
    instantes garante que todas as regras criadas ou fundidas mantêm os pesos transformados
    sem copiar arrays a cada amostra.
    """

    def __init__(self, base_model, mode: str = "FULL", seed: int = 0):
        assert mode in {"FULL", "NO_W", "SHUFFLED_W"}, f"Modo inválido: {mode}"
        self.model = base_model
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.model.n_features)
        self._ones = np.ones(self.model.n_features, dtype=float)
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

    def _transform_vec(self, w: np.ndarray) -> np.ndarray:
        if self.mode == "NO_W":
            return self._ones.copy()
        if self.mode == "SHUFFLED_W":
            return np.asarray(w, dtype=float)[self.perm].copy()
        return w

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