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
        n = X.shape[0]
        preds = np.empty(n, dtype=int)
        curr = self.last_y
        for i in range(n):
            preds[i] = curr
        return preds

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=int)
        if y_arr.size > 0:
            self.last_y = int(y_arr[-1])


class AblationWeightWrapper:
    """
    Wrapper não-intrusivo sobre EvolvingNeuroFuzzyUnimSafe.
    Nunca altera ENF_UnimNeuron_Safe.py:
      - FULL: mantém os pesos aprendidos intactos.
      - NO_W: coloca r.w = 1.0 em todas as regras (testa escala de pesos).
      - SHUFFLED_W: permuta r.w segundo uma permutação aleatória fixa (testa conteúdo de informação).
    """

    def __init__(self, base_model, mode: str = "FULL", seed: int = 0):
        assert mode in {"FULL", "NO_W", "SHUFFLED_W"}, f"Modo inválido: {mode}"
        self.model = base_model
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.model.n_features)

        # Estado sombra para evitar dupla permutação durante merges internos
        self._clean_global_w = self.model.current_feature_weights.copy()
        self._clean_rule_w = {id(r): r.w.copy() for r in self.model.rules}

    @property
    def n_rules_(self) -> int:
        return self.model.n_rules_

    @property
    def rules(self):
        return self.model.rules

    @property
    def current_feature_weights(self) -> np.ndarray:
        return self.model.current_feature_weights

    def _restore_clean_weights(self) -> None:
        if self.mode == "FULL":
            return
        self.model.current_feature_weights = self._clean_global_w.copy()
        for r in self.model.rules:
            if id(r) in self._clean_rule_w:
                r.w = self._clean_rule_w[id(r)].copy()

    def _snapshot_and_apply_ablation(self) -> None:
        if self.mode == "FULL":
            return

        # 1. Guardar pesos genuínos após o update
        self._clean_global_w = self.model.current_feature_weights.copy()
        alive_ids = set()
        for r in self.model.rules:
            rid = id(r)
            alive_ids.add(rid)
            self._clean_rule_w[rid] = r.w.copy()

        # Limpar regras antigas removidas
        for old_id in list(self._clean_rule_w.keys()):
            if old_id not in alive_ids:
                del self._clean_rule_w[old_id]

        # 2. Aplicar transformação (NO_W ou SHUFFLED_W) para as próximas predições
        if self.mode == "NO_W":
            self.model.current_feature_weights = np.ones_like(self._clean_global_w)
            for r in self.model.rules:
                r.w = np.ones_like(r.w)
        elif self.mode == "SHUFFLED_W":
            self.model.current_feature_weights = self._clean_global_w[self.perm].copy()
            for r in self.model.rules:
                r.w = self._clean_rule_w[id(r)][self.perm].copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        for xi, yi in zip(X, y):
            self._restore_clean_weights()
            self.model.partial_fit(xi.reshape(1, -1), np.array([int(yi)], dtype=int))
            self._snapshot_and_apply_ablation()