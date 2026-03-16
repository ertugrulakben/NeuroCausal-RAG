class CausalStrengthCalculator:
    def calculate_weight(self, semantic_sim: float, temporal_dist: float, nli_score: float) -> float:
        """
        Calculates the causal weight based on the formula:
        Weight = (Semantic Similarity * 0.4) + (Temporal Distance * 0.3) + (NLI Entailment Score * 0.3)
        
        Args:
            semantic_sim: Float between 0.0 and 1.0
            temporal_dist: Normalized temporal distance score (1.0 = close, 0.0 = far)
            nli_score: Entailment score (0.0 to 1.0)
        """
        # Ensure inputs are within bounds
        semantic_sim = max(0.0, min(1.0, semantic_sim))
        temporal_dist = max(0.0, min(1.0, temporal_dist))
        nli_score = max(0.0, min(1.0, nli_score))
        
        weight = (semantic_sim * 0.4) + (temporal_dist * 0.3) + (nli_score * 0.3)
        return round(weight, 4)
