import numpy as np

from ENF_UnimNeuron_Safe import (
    EvolvingNeuroFuzzyUnimSafe,
    FeatureSeparabilityBuffer
)


class LocalUnimNeuron(EvolvingNeuroFuzzyUnimSafe):

    def __init__(self, *args, local_window=100, **kwargs):

        super().__init__(*args, **kwargs)

        self.local_window = local_window
        self.local_buffers = {}

    def _update_one(self, x, y):

        super()._update_one(x, y)

        if not self.rules:
            return

        # Remove buffers belonging to deleted rules
        active_ids = {id(rule) for rule in self.rules}

        self.local_buffers = {
            key: value
            for key, value in self.local_buffers.items()
            if key in active_ids
        }

        # Initialize buffers for new rules
        for rule in self.rules:

            key = id(rule)

            if key not in self.local_buffers:

                self.local_buffers[key] = (
                    rule,
                    FeatureSeparabilityBuffer(
                        self.n_features,
                        self.local_window
                    )
                )

        # Assign the observation to the nearest surviving rule
        x = np.asarray(x).ravel()

        distances = [
            np.max(np.abs(x - rule.center))
            for rule in self.rules
        ]

        nearest_rule = self.rules[np.argmin(distances)]

        buffer = self.local_buffers[id(nearest_rule)][1]

        buffer.add(x, y)

        # Update local relevance every 50 samples
        if self.K % 50 == 0:

            for rule in self.rules:

                buffer = self.local_buffers[id(rule)][1]

                _, counts = np.unique(
                    buffer.y,
                    return_counts=True
                )

                if len(counts) >= 2 and min(counts) >= 10:

                    rule.w = buffer.compute_feature_weights()

                else:

                    rule.w = (
                        self.current_feature_weights.copy()
                    )