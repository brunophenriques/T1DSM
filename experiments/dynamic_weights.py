from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


class DynamicUnimNeuron(EvolvingNeuroFuzzyUnimSafe):

    def _update_one(self, x, y):

        super()._update_one(x, y)

        if self.K % 50 == 0:

            for rule in self.rules:

                rule.w = self.current_feature_weights.copy()