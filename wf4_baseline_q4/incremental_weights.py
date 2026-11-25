import numpy as np
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


class IncrementalSeparabilityBuffer:
    """
    Estimador incremental compatível com a interface de FeatureSeparabilityBuffer
    (métodos add(x, y) e compute_feature_weights()), substituindo o buffer de 200
    amostras por estatísticas online O(C * d) por classe e feature (alpha=0.05).
    """

    def __init__(
        self,
        n_features: int,
        method: str = "INC_FISHER_L2",
        alpha: float = 0.05,
        eps: float = 1e-6,
    ):
        assert method in {"INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"}, (
            f"Método incremental inválido: {method}"
        )
        self.n_features = int(n_features)
        self.method = method
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.n_samples = 0
        self.class_means: dict[int, np.ndarray] = {}
        self.class_vars: dict[int, np.ndarray] = {}
        self.relieff_scores = np.zeros(self.n_features, dtype=float)

    def add(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=float).ravel()
        y = int(y)
        self.n_samples += 1

        if y not in self.class_means:
            self.class_means[y] = x.copy()
            self.class_vars[y] = np.zeros(self.n_features, dtype=float)
            return

        # Para INC_RELIEFF, calcula margem nearest-miss vs nearest-hit antes de atualizar
        if self.method == "INC_RELIEFF" and len(self.class_means) >= 2:
            d_hit = np.abs(x - self.class_means[y])
            other_classes = [c for c in self.class_means if c != y]
            nearest_miss = min(
                (np.abs(x - self.class_means[c]) for c in other_classes),
                key=lambda d: float(np.sum(d)),
            )
            margin = nearest_miss - d_hit
            self.relieff_scores = (1.0 - self.alpha) * self.relieff_scores + self.alpha * margin

        diff = x - self.class_means[y]
        self.class_means[y] = self.class_means[y] + self.alpha * diff
        self.class_vars[y] = (1.0 - self.alpha) * (
            self.class_vars[y] + self.alpha * (diff ** 2)
        )

    def _normalize_like_baseline(self, raw_sep: np.ndarray) -> np.ndarray:
        sep = np.nan_to_num(np.asarray(raw_sep, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        sep = np.maximum(sep, 0.0)
        if np.all(sep <= 0.0):
            return np.ones(self.n_features, dtype=float) / float(self.n_features)
        sep_norm = sep / sep.max()
        w = 0.2 + 0.8 * sep_norm
        return w / w.max()

    def compute_feature_weights(self) -> np.ndarray:
        classes = list(self.class_means.keys())
        if self.n_samples < 10 or len(classes) < 2:
            return np.ones(self.n_features, dtype=float) / float(self.n_features)

        if self.method == "INC_FISHER_L2":
            between = np.zeros(self.n_features, dtype=float)
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    between += (self._class_diff(classes[i], classes[j])) ** 2
            within = sum(self.class_vars[c] for c in classes) + self.eps
            return self._normalize_like_baseline(between / within)

        if self.method == "INC_L1_CENTROID":
            l1_sep = np.zeros(self.n_features, dtype=float)
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    l1_sep += np.abs(self._class_diff(classes[i], classes[j]))
            return self._normalize_like_baseline(l1_sep)

        return self._normalize_like_baseline(self.relieff_scores)

    def _class_diff(self, c1: int, c2: int) -> np.ndarray:
        return self.class_means[c1] - self.class_means[c2]


class ENFUnimNeuronIncremental(EvolvingNeuroFuzzyUnimSafe):
    """
    Substitui self.sep_buffer (FeatureSeparabilityBuffer de 200 amostras) por
    IncrementalSeparabilityBuffer sem executar o buffer original em paralelo.
    Mantém a política original de criação/fusão e a cadência de 50 observações
    (K % 50 == 0) para isolar estritamente o efeito do estimador de relevância.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        inc_method: str = "INC_FISHER_L2",
        alpha_inc: float = 0.05,
        update_rule_weights: bool = False,
        **kwargs,
    ):
        super().__init__(n_features=n_features, n_classes=n_classes, **kwargs)
        self.inc_method = inc_method
        self.alpha_inc = float(alpha_inc)
        self.update_rule_weights = bool(update_rule_weights)

        # Substitui o buffer original (não o mantém em paralelo)
        self.sep_buffer = IncrementalSeparabilityBuffer(
            n_features=self.n_features,
            method=self.inc_method,
            alpha=self.alpha_inc,
        )

    def _update_one(self, x: np.ndarray, y: int) -> None:
        super()._update_one(x, y)
        if self.update_rule_weights and self.K % 50 == 0:
            w_now = self.current_feature_weights.copy()
            for r in self.rules:
                r.w = w_now.copy()