from datetime import datetime
import re

class TemporalEngine:
    def extract_date(self, text: str):
        # Simple regex for YYYY-MM-DD or YYYY
        match = re.search(r'\d{4}-\d{2}-\d{2}', text)
        if match:
            return datetime.strptime(match.group(), '%Y-%m-%d')
        match = re.search(r'\d{4}', text)
        if match:
            return datetime.strptime(match.group(), '%Y')
        return None

    def validate_causal_order(self, cause_text: str, effect_text: str) -> bool:
        """
        Returns True if cause happens before effect or if dates are unknown.
        Returns False if cause happens after effect.
        """
        date_cause = self.extract_date(cause_text)
        date_effect = self.extract_date(effect_text)
        
        if date_cause and date_effect:
            return date_cause <= date_effect
            
        return True
