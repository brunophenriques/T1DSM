import numpy as np

from ENF_UnimNeuron_Safe import (
    EvolvingNeuroFuzzyUnimSafe
)

from experiments.incremental_relevance import (
    IncrementalRelevance
)


class IncrementalLocalUnimNeuron(
    EvolvingNeuroFuzzyUnimSafe
):

    def __init__(
        self,
        *args,
        forgetting=0.98,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.forgetting = forgetting
        self.local_estimators = {}

    def _update_one(self, x, y):

        super()._update_one(x, y)

        if not self.rules:
            return

        active_ids = {
            id(rule) for rule in self.rules
        }

        self.local_estimators = {
            key: value
            for key, value in
            self.local_estimators.items()
            if key in active_ids
        }

        for rule in self.rules:

            key = id(rule)

            if key not in self.local_estimators:

                self.local_estimators[key] = (
                    rule,
                    IncrementalRelevance(
                        self.n_features,
                        self.n_classes,
                        self.forgetting
                    )
                )

        x = np.asarray(x).ravel()

        distances = [
            np.max(np.abs(x - rule.center))
            for rule in self.rules
        ]

        nearest_rule = self.rules[
            np.argmin(distances)
        ]

        estimator = self.local_estimators[
            id(nearest_rule)
        ][1]

        estimator.update(x, y)

        if self.K % 50 == 0:

            for rule in self.rules:

                estimator = self.local_estimators[
                    id(rule)
                ][1]

                weights = estimator.compute_weights()

                if weights is None:

                    rule.w = (
                        self.current_feature_weights.copy()
                    )

                else:

                    rule.w = weights