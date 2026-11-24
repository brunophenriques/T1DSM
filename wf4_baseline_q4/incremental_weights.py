import numpy as np
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


class ENFUnimNeuronIncremental(EvolvingNeuroFuzzyUnimSafe):
    """
    Subclasse de EvolvingNeuroFuzzyUnimSafe que substitui o refresh discreto
    de 50 amostras do FeatureSeparabilityBuffer por uma atualização incremental
    amostra a amostra (INC_FISHER_L2, INC_L1_CENTROID ou INC_RELIEFF).
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
        assert inc_method in {"INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"}, (
            f"Método incremental inválido: {inc_method}"
        )
        self.inc_method = inc_method
        self.alpha_inc = float(alpha_inc)
        self.update_rule_weights = bool(update_rule_weights)

        # Estatísticas incrementais por classe (sem janela de amostras passadas)
        self._class_means = {}
        self._class_vars = {}
        self._relieff_scores = np.zeros(self.n_features, dtype=float)

    def _normalize_like_baseline(self, raw_sep: np.ndarray) -> np.ndarray:
        """Aplica exatamente a mesma escala [0.2, 1.0] do FeatureSeparabilityBuffer original."""
        sep = np.maximum(np.asarray(raw_sep, dtype=float), 0.0)
        if np.all(sep <= 1e-12):
            return np.ones(self.n_features, dtype=float) / float(self.n_features)
        sep_norm = sep / sep.max()
        w = 0.2 + 0.8 * sep_norm
        return w / w.max()

    def _step_incremental_weights(self, x: np.ndarray, y: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        y = int(y)

        if y not in self._class_means:
            self._class_means[y] = x.copy()
            self._class_vars[y] = np.ones(self.n_features, dtype=float) * 1e-3
        else:
            diff = x - self._class_means[y]
            self._class_means[y] = self._class_means[y] + self.alpha_inc * diff
            self._class_vars[y] = (1.0 - self.alpha_inc) * (
                self._class_vars[y] + self.alpha_inc * (diff ** 2)
            )

        classes = list(self._class_means.keys())
        if len(classes) < 2:
            return np.ones(self.n_features, dtype=float) / float(self.n_features)

        if self.inc_method == "INC_FISHER_L2":
            between = np.zeros(self.n_features, dtype=float)
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    between += (self._class_means[classes[i]] - self._class_means[classes[j]]) ** 2
            within = sum(self._class_vars[c] for c in classes) + 1e-6
            return self._normalize_like_baseline(between / within)

        if self.inc_method == "INC_L1_CENTROID":
            l1_sep = np.zeros(self.n_features, dtype=float)
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    l1_sep += np.abs(self._class_means[classes[i]] - self._class_means[classes[j]])
            return self._normalize_like_baseline(l1_sep)

        # INC_RELIEFF: margem local entre nearest-miss e nearest-hit
        d_hit = np.abs(x - self._class_means[y])
        other_classes = [c for c in classes if c != y]
        d_misses = [np.abs(x - self._class_means[c]) for c in other_classes]
        nearest_miss = min(d_misses, key=lambda d: float(np.sum(d)))

        margin = nearest_miss - d_hit
        self._relieff_scores = (1.0 - self.alpha_inc) * self._relieff_scores + self.alpha_inc * margin
        return self._normalize_like_baseline(self._relieff_scores)

    def _update_one(self, x: np.ndarray, y: int) -> None:
        # 1. Executa a evolução normal das regras do UnimNeuron
        super()._update_one(x, y)

        # 2. Sobrepõe o vetor global (que a superclasse só atualiza a cada 50 passos)
        #    pelo cálculo incremental contínuo a cada amostra t
        new_w = self._step_incremental_weights(x, y)
        self.current_feature_weights = new_w.copy()

        if self.update_rule_weights:
            for r in self.rules:
                r.w = new_w.copy()