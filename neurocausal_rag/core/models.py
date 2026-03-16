from pydantic import BaseModel
from typing import Optional, List, Any, Dict

class ActionRequired(BaseModel):
    type: str = "human_intervention"
    message: str
    options: List[str]
    context: Any

class SearchResult(BaseModel):
    node_id: Optional[str] = None
    content: Optional[str] = None 
    text: Optional[str] = None # For backward compatibility if needed
    score: float
    similarity_score: Optional[float] = 0.0
    causal_score: Optional[float] = 0.0
    importance_score: Optional[float] = 0.0
    causal_chain: Optional[List[str]] = None
    metadata: Dict = {}
    resolved_entities: Optional[List[str]] = None
    action_required: Optional[ActionRequired] = None
