"""
NeuroCausal RAG - Agent Tools
LangGraph/LangChain compatible tool definitions

Tools:
- SearchTool: Semantic + causal search
- GraphTool: Graph navigation and exploration
- VerifyTool: Fact verification against sources

Yazar: Ertuğrul Akben
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard tool execution result"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class BaseTool:
    """Base class for all agent tools"""

    name: str = "base_tool"
    description: str = "Base tool"

    def __init__(self):
        self._call_count = 0

    def __call__(self, *args, **kwargs) -> ToolResult:
        self._call_count += 1
        try:
            result = self.execute(*args, **kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.error(f"Tool {self.name} error: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def to_langchain_tool(self):
        """Convert to LangChain Tool format"""
        try:
            from langchain.tools import Tool
            return Tool(
                name=self.name,
                description=self.description,
                func=lambda x: self.execute(x)
            )
        except ImportError:
            logger.warning("langchain not installed")
            return None

    def to_openai_function(self) -> Dict:
        """Convert to OpenAI function calling format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }

    def get_parameters_schema(self) -> Dict:
        """Override in subclasses to define parameters"""
        return {"type": "object", "properties": {}}


class SearchTool(BaseTool):
    """
    Semantic + Causal search tool.

    Searches the knowledge base using hybrid scoring:
    - Semantic similarity (embedding)
    - Causal relevance (graph connections)
    - Importance score (PageRank)
    """

    name = "search"
    description = """Search the knowledge base for relevant documents.
    Use this tool when you need to find information about a topic.
    Input should be a natural language query.
    Returns ranked documents with causal chains."""

    def __init__(self, retriever, top_k: int = 5):
        super().__init__()
        self.retriever = retriever
        self.top_k = top_k

    def execute(self, query: str, **kwargs) -> Dict:
        """Execute search query"""
        top_k = kwargs.get('top_k', self.top_k)
        alpha = kwargs.get('alpha')
        beta = kwargs.get('beta')
        gamma = kwargs.get('gamma')

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        formatted = []
        for r in results:
            formatted.append({
                'id': r.node_id,
                'content': r.content[:500],
                'score': r.score,
                'causal_chain': r.causal_chain,
                'is_injected': r.metadata.get('injected_from') if r.metadata else None
            })

        return {
            'query': query,
            'results': formatted,
            'count': len(formatted)
        }

    def get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }


class GraphTool(BaseTool):
    """
    Graph navigation and exploration tool.

    Allows the agent to:
    - Get node details
    - Find causal chains
    - Explore neighbors
    - Find paths between nodes
    """

    name = "graph"
    description = """Navigate the causal knowledge graph.
    Use this tool to explore relationships between documents,
    find causal chains, or understand document connections.
    Actions: get_node, get_chain, get_neighbors, find_path"""

    def __init__(self, graph_engine):
        super().__init__()
        self.graph = graph_engine

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute graph operation"""
        if action == "get_node":
            return self._get_node(kwargs.get('node_id'))
        elif action == "get_chain":
            return self._get_chain(
                kwargs.get('node_id'),
                kwargs.get('max_depth', 3),
                kwargs.get('direction', 'forward')
            )
        elif action == "get_neighbors":
            return self._get_neighbors(
                kwargs.get('node_id'),
                kwargs.get('relation_types')
            )
        elif action == "find_path":
            return self._find_path(
                kwargs.get('source_id'),
                kwargs.get('target_id')
            )
        elif action == "stats":
            return self._get_stats()
        else:
            return {"error": f"Unknown action: {action}"}

    def _get_node(self, node_id: str) -> Dict:
        node = self.graph.get_node(node_id)
        if node:
            return {
                'id': node['id'],
                'content': node['content'][:500],
                'importance': node.get('importance', 0),
                'metadata': node.get('metadata', {})
            }
        return {"error": f"Node not found: {node_id}"}

    def _get_chain(self, node_id: str, max_depth: int, direction: str) -> Dict:
        chain = self.graph.get_causal_chain(node_id, max_depth, direction)
        chain_details = []
        for nid in chain:
            node = self.graph.get_node(nid)
            if node:
                chain_details.append({
                    'id': nid,
                    'content': node['content'][:200]
                })
        return {
            'start_node': node_id,
            'direction': direction,
            'chain': chain,
            'chain_details': chain_details,
            'length': len(chain)
        }

    def _get_neighbors(self, node_id: str, relation_types: List[str] = None) -> Dict:
        outgoing = self.graph.get_neighbors(node_id, relation_types)
        incoming = self.graph.get_predecessors(node_id, relation_types)
        return {
            'node_id': node_id,
            'outgoing': outgoing,
            'incoming': incoming,
            'total_connections': len(outgoing) + len(incoming)
        }

    def _find_path(self, source_id: str, target_id: str) -> Dict:
        path, score = self.graph.find_causal_path(source_id, target_id)
        path_details = []
        for nid in path:
            node = self.graph.get_node(nid)
            if node:
                path_details.append({
                    'id': nid,
                    'content': node['content'][:200]
                })
        return {
            'source': source_id,
            'target': target_id,
            'path': path,
            'path_details': path_details,
            'score': score,
            'connected': len(path) > 0
        }

    def _get_stats(self) -> Dict:
        return self.graph.get_stats()

    def get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_node", "get_chain", "get_neighbors", "find_path", "stats"],
                    "description": "The graph operation to perform"
                },
                "node_id": {
                    "type": "string",
                    "description": "The node ID for operations"
                },
                "source_id": {
                    "type": "string",
                    "description": "Source node for path finding"
                },
                "target_id": {
                    "type": "string",
                    "description": "Target node for path finding"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum chain depth",
                    "default": 3
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward"],
                    "default": "forward"
                }
            },
            "required": ["action"]
        }


class VerifyTool(BaseTool):
    """
    Fact verification tool.

    Verifies claims against the knowledge base:
    - Checks if claim is supported by sources
    - Finds contradicting evidence
    - Calculates confidence score
    """

    name = "verify"
    description = """Verify a claim against the knowledge base.
    Use this tool to check if a statement is supported by evidence.
    Returns supporting/contradicting documents and confidence score."""

    def __init__(self, retriever, llm_client=None):
        super().__init__()
        self.retriever = retriever
        self.llm_client = llm_client

    def execute(self, claim: str, **kwargs) -> Dict:
        """Verify a claim"""
        # Search for relevant documents
        results = self.retriever.search(claim, top_k=5)

        if not results:
            return {
                'claim': claim,
                'verified': False,
                'confidence': 0.0,
                'reason': 'No relevant documents found'
            }

        # Collect evidence
        supporting = []
        contradicting = []

        for r in results:
            # Simple heuristic: high score = supporting
            if r.score > 0.7:
                supporting.append({
                    'id': r.node_id,
                    'content': r.content[:300],
                    'score': r.score
                })
            elif r.score < 0.3:
                contradicting.append({
                    'id': r.node_id,
                    'content': r.content[:300],
                    'score': r.score
                })

        # Calculate confidence
        if supporting:
            avg_support = sum(s['score'] for s in supporting) / len(supporting)
        else:
            avg_support = 0.0

        confidence = min(1.0, avg_support * (len(supporting) / 3))

        return {
            'claim': claim,
            'verified': len(supporting) > 0,
            'confidence': confidence,
            'supporting_evidence': supporting[:3],
            'contradicting_evidence': contradicting[:2],
            'evidence_count': len(results)
        }

    def get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The claim to verify"
                }
            },
            "required": ["claim"]
        }


class ReasonTool(BaseTool):
    """
    Multi-step reasoning tool.

    Uses LLM to perform complex reasoning:
    - Causal inference
    - Multi-hop questions
    - Synthesis of multiple sources
    """

    name = "reason"
    description = """Perform complex reasoning over information.
    Use this for multi-hop questions or causal inference.
    Requires context from previous search results."""

    def __init__(self, llm_client):
        super().__init__()
        self.llm_client = llm_client

    def execute(self, question: str, context: List[Dict], **kwargs) -> Dict:
        """Perform reasoning"""
        if not self.llm_client:
            return {
                'error': 'LLM client not configured',
                'answer': None
            }

        # Build prompt
        context_text = "\n\n".join([
            f"[{i+1}] {doc.get('content', doc)[:400]}"
            for i, doc in enumerate(context[:5])
        ])

        prompt = f"""Based on the following context, answer the question.
Think step by step and cite relevant sources.

Context:
{context_text}

Question: {question}

Answer:"""

        try:
            response = self.llm_client.generate(prompt)
            return {
                'question': question,
                'answer': response,
                'context_used': len(context),
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'success': False
            }

    def get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to reason about"
                },
                "context": {
                    "type": "array",
                    "description": "Context documents from search",
                    "items": {"type": "object"}
                }
            },
            "required": ["question", "context"]
        }


def create_tools(
    retriever=None,
    graph_engine=None,
    llm_client=None
) -> Dict[str, BaseTool]:
    """
    Factory function to create all available tools.

    Args:
        retriever: NeuroCausal retriever instance
        graph_engine: Graph engine instance
        llm_client: LLM client instance

    Returns:
        Dictionary of tool name -> tool instance
    """
    tools = {}

    if retriever:
        tools['search'] = SearchTool(retriever)
        tools['verify'] = VerifyTool(retriever, llm_client)

    if graph_engine:
        tools['graph'] = GraphTool(graph_engine)

    if llm_client:
        tools['reason'] = ReasonTool(llm_client)

    return tools
