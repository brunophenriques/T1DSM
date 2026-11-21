import numpy as np


class IncrementalRelevance:

    def __init__(
        self,
        n_features,
        n_classes=2,
        forgetting=0.98
    ):

        self.forgetting = forgetting

        self.count = np.zeros(n_classes)

        self.sum = np.zeros(
            (n_classes, n_features)
        )

        self.sum_sq = np.zeros(
            (n_classes, n_features)
        )

    def update(self, x, y):

        x = np.asarray(x, dtype=float).ravel()

        # Exponential forgetting
        self.count *= self.forgetting
        self.sum *= self.forgetting
        self.sum_sq *= self.forgetting

        # Incorporate the new observation
        self.count[y] += 1
        self.sum[y] += x
        self.sum_sq[y] += x ** 2

    def compute_weights(self):

        if np.any(self.count < 5):
            return None

        total = self.count.sum()

        means = self.sum / self.count[:, None]

        global_mean = self.sum.sum(axis=0) / total

        between = np.sum(
            self.count[:, None]
            * (means - global_mean) ** 2,
            axis=0
        )

        within = np.sum(
            np.maximum(
                self.sum_sq
                - self.sum ** 2 / self.count[:, None],
                0
            ),
            axis=0
        )

        relevance = between / (
            between + within + 1e-8
        )

        return 0.2 + 0.8 * np.clip(
            relevance, 0, 1
        )