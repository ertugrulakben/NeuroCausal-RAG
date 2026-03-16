"""
NeuroCausal RAG - LangGraph Agent
Self-correcting multi-step reasoning agent

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     CausalRAGAgent                          │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
    │  │ Analyze │ -> │ Retrieve│ -> │ Reason  │ -> │ Verify  │  │
    │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
    │       │              │              │              │        │
    │       └──────────────┴──────────────┴──────────────┘        │
    │                         │                                   │
    │                    Self-Correct Loop                        │
    │                   (if confidence < 0.7)                     │
    └─────────────────────────────────────────────────────────────┘

Yazar: Ertuğrul Akben
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import operator
import logging

logger = logging.getLogger(__name__)


class AgentStep(str, Enum):
    """Agent execution steps"""
    ANALYZE = "analyze"
    RETRIEVE = "retrieve"
    REASON = "reason"
    VERIFY = "verify"
    CORRECT = "correct"
    COMPLETE = "complete"


class AgentState(TypedDict):
    """
    Agent state for LangGraph.

    Tracks the entire reasoning process.
    """
    # Input
    query: str
    context: Optional[Dict]

    # Execution
    current_step: str
    steps_executed: List[str]
    iteration: int
    max_iterations: int

    # Results
    search_results: List[Dict]
    graph_explorations: List[Dict]
    reasoning_chain: List[str]
    answer: Optional[str]

    # Quality
    confidence: float
    verified: bool
    corrections_made: int

    # Metadata
    tool_calls: List[Dict]
    errors: List[str]


def create_initial_state(query: str, max_iterations: int = 3) -> AgentState:
    """Create initial agent state"""
    return AgentState(
        query=query,
        context=None,
        current_step=AgentStep.ANALYZE.value,
        steps_executed=[],
        iteration=0,
        max_iterations=max_iterations,
        search_results=[],
        graph_explorations=[],
        reasoning_chain=[],
        answer=None,
        confidence=0.0,
        verified=False,
        corrections_made=0,
        tool_calls=[],
        errors=[]
    )


class CausalRAGAgent:
    """
    Self-correcting Causal RAG Agent.

    Uses a state graph for multi-step reasoning with:
    - Semantic + causal search
    - Graph-based exploration
    - LLM reasoning
    - Self-verification and correction

    Can run with or without LangGraph (fallback to simple execution).

    Example:
        >>> agent = CausalRAGAgent(retriever, graph, llm)
        >>> result = agent.run("What causes global warming?")
        >>> print(result['answer'])
    """

    def __init__(
        self,
        retriever=None,
        graph_engine=None,
        llm_client=None,
        min_confidence: float = 0.7,
        max_iterations: int = 3,
        use_langgraph: bool = True
    ):
        """
        Args:
            retriever: NeuroCausal retriever
            graph_engine: Graph engine
            llm_client: LLM client
            min_confidence: Minimum confidence to accept answer
            max_iterations: Maximum self-correction iterations
            use_langgraph: Try to use LangGraph if available
        """
        self.retriever = retriever
        self.graph = graph_engine
        self.llm = llm_client
        self.min_confidence = min_confidence
        self.max_iterations = max_iterations

        # Try to use LangGraph
        self._langgraph_available = False
        if use_langgraph:
            try:
                from langgraph.graph import StateGraph
                self._langgraph_available = True
                self._graph = self._build_langgraph()
                logger.info("CausalRAGAgent: Using LangGraph")
            except ImportError:
                logger.info("CausalRAGAgent: LangGraph not available, using simple execution")

    def _build_langgraph(self):
        """Build LangGraph state graph"""
        from langgraph.graph import StateGraph, END

        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze", self._step_analyze)
        workflow.add_node("retrieve", self._step_retrieve)
        workflow.add_node("reason", self._step_reason)
        workflow.add_node("verify", self._step_verify)
        workflow.add_node("correct", self._step_correct)

        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "verify")

        # Conditional edge from verify
        workflow.add_conditional_edges(
            "verify",
            self._should_correct,
            {
                "correct": "correct",
                "end": END
            }
        )

        # Correction loops back to retrieve
        workflow.add_edge("correct", "retrieve")

        return workflow.compile()

    def run(self, query: str, context: Dict = None) -> Dict:
        """
        Run the agent on a query.

        Args:
            query: User question
            context: Optional additional context

        Returns:
            Result dictionary with answer, confidence, reasoning chain
        """
        state = create_initial_state(query, self.max_iterations)
        if context:
            state['context'] = context

        if self._langgraph_available:
            # Use LangGraph
            final_state = self._graph.invoke(state)
        else:
            # Simple fallback execution
            final_state = self._simple_execution(state)

        return {
            'query': query,
            'answer': final_state.get('answer'),
            'confidence': final_state.get('confidence', 0.0),
            'verified': final_state.get('verified', False),
            'reasoning_chain': final_state.get('reasoning_chain', []),
            'search_results': final_state.get('search_results', []),
            'iterations': final_state.get('iteration', 0),
            'corrections': final_state.get('corrections_made', 0),
            'tool_calls': final_state.get('tool_calls', [])
        }

    def _simple_execution(self, state: AgentState) -> AgentState:
        """Simple execution without LangGraph"""
        while state['iteration'] < state['max_iterations']:
            state['iteration'] += 1

            # Step 1: Analyze
            state = self._step_analyze(state)

            # Step 2: Retrieve
            state = self._step_retrieve(state)

            # Step 3: Reason
            state = self._step_reason(state)

            # Step 4: Verify
            state = self._step_verify(state)

            # Check if done
            if state['confidence'] >= self.min_confidence or state['verified']:
                break

            # Correct and retry
            state = self._step_correct(state)

        return state

    def _step_analyze(self, state: AgentState) -> AgentState:
        """Analyze the query to determine search strategy"""
        state['steps_executed'].append(AgentStep.ANALYZE.value)
        state['current_step'] = AgentStep.RETRIEVE.value

        query = state['query']

        # Simple query analysis
        analysis = {
            'query': query,
            'is_causal': any(w in query.lower() for w in ['neden', 'cause', 'why', 'nasıl', 'how']),
            'is_comparison': any(w in query.lower() for w in ['karşılaştır', 'compare', 'fark', 'difference']),
            'is_factual': any(w in query.lower() for w in ['ne', 'what', 'when', 'where', 'kim', 'who']),
            'needs_chain': any(w in query.lower() for w in ['zincir', 'chain', 'bağlantı', 'connection'])
        }

        state['reasoning_chain'].append(f"Query analysis: {analysis}")
        state['tool_calls'].append({'tool': 'analyze', 'input': query, 'output': analysis})

        return state

    def _step_retrieve(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents"""
        state['steps_executed'].append(AgentStep.RETRIEVE.value)
        state['current_step'] = AgentStep.REASON.value

        if not self.retriever:
            state['errors'].append("Retriever not configured")
            return state

        query = state['query']

        # Adjust search based on iteration (more causal on retries)
        iteration = state['iteration']
        if iteration > 1:
            # Increase causal weight on retries
            alpha = max(0.3, 0.5 - (iteration * 0.1))
            beta = min(0.5, 0.3 + (iteration * 0.1))
            gamma = 0.2
        else:
            alpha, beta, gamma = None, None, None

        try:
            results = self.retriever.search(query, top_k=5, alpha=alpha, beta=beta, gamma=gamma)

            search_results = []
            for r in results:
                search_results.append({
                    'id': r.node_id,
                    'content': r.content,
                    'score': r.score,
                    'chain': r.causal_chain
                })

            state['search_results'] = search_results
            state['reasoning_chain'].append(f"Retrieved {len(results)} documents")
            state['tool_calls'].append({
                'tool': 'search',
                'input': query,
                'output': f"{len(results)} results"
            })

            # Explore graph if results have chains
            if self.graph and any(r.get('chain') for r in search_results):
                explorations = []
                for r in search_results[:2]:
                    if r.get('chain') and len(r['chain']) > 1:
                        chain_id = r['chain'][0]
                        chain = self.graph.get_causal_chain(chain_id, max_depth=3)
                        explorations.append({
                            'start': chain_id,
                            'chain': chain
                        })
                state['graph_explorations'] = explorations
                state['reasoning_chain'].append(f"Explored {len(explorations)} causal chains")

        except Exception as e:
            state['errors'].append(f"Retrieve error: {e}")

        return state

    def _step_reason(self, state: AgentState) -> AgentState:
        """Generate answer using LLM"""
        state['steps_executed'].append(AgentStep.REASON.value)
        state['current_step'] = AgentStep.VERIFY.value

        if not self.llm:
            # Simple answer without LLM
            if state['search_results']:
                top_result = state['search_results'][0]
                state['answer'] = top_result['content'][:500]
                state['confidence'] = top_result['score']
            else:
                state['answer'] = "Bilgi bulunamadı."
                state['confidence'] = 0.0
            return state

        # Build context from search results
        context_parts = []
        for i, r in enumerate(state['search_results'][:5], 1):
            ctx = f"[{i}] {r['content'][:400]}"
            if r.get('chain') and len(r['chain']) > 1:
                ctx += f"\n    Causal chain: {' → '.join(r['chain'][:3])}"
            context_parts.append(ctx)

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based on provided context.
Use the causal chains to explain relationships when relevant.

Context:
{context}

Question: {state['query']}

Instructions:
1. Answer based on the context provided
2. Mention causal relationships if relevant
3. Cite sources using [1], [2], etc.
4. Be concise but complete

Answer:"""

        try:
            answer = self.llm.generate(prompt)
            state['answer'] = answer
            state['reasoning_chain'].append("Generated answer using LLM")
            state['tool_calls'].append({
                'tool': 'llm',
                'input': 'reasoning prompt',
                'output': 'answer generated'
            })
        except Exception as e:
            state['errors'].append(f"Reason error: {e}")
            if state['search_results']:
                state['answer'] = state['search_results'][0]['content'][:500]

        return state

    def _step_verify(self, state: AgentState) -> AgentState:
        """Verify the answer quality"""
        state['steps_executed'].append(AgentStep.VERIFY.value)
        state['current_step'] = AgentStep.COMPLETE.value

        if not state['answer']:
            state['confidence'] = 0.0
            state['verified'] = False
            return state

        # Calculate confidence based on multiple factors
        factors = []

        # 1. Search result quality
        if state['search_results']:
            avg_score = sum(r['score'] for r in state['search_results']) / len(state['search_results'])
            factors.append(avg_score)
        else:
            factors.append(0.0)

        # 2. Answer length (too short = low confidence)
        answer_len = len(state['answer'])
        if answer_len < 50:
            factors.append(0.3)
        elif answer_len < 200:
            factors.append(0.6)
        else:
            factors.append(0.8)

        # 3. Causal chain usage
        if state['graph_explorations']:
            factors.append(0.8)
        else:
            factors.append(0.5)

        # 4. No errors
        if not state['errors']:
            factors.append(0.9)
        else:
            factors.append(0.4)

        # Calculate final confidence
        state['confidence'] = sum(factors) / len(factors)

        # Verify
        state['verified'] = state['confidence'] >= self.min_confidence

        state['reasoning_chain'].append(
            f"Verification: confidence={state['confidence']:.2f}, verified={state['verified']}"
        )

        return state

    def _step_correct(self, state: AgentState) -> AgentState:
        """Self-correct by adjusting strategy"""
        state['steps_executed'].append(AgentStep.CORRECT.value)
        state['current_step'] = AgentStep.RETRIEVE.value
        state['corrections_made'] += 1

        # Analyze what went wrong
        if not state['search_results']:
            # No results - try broader query
            words = state['query'].split()
            if len(words) > 3:
                state['query'] = ' '.join(words[:3])
                state['reasoning_chain'].append("Correction: Simplified query")
        elif state['confidence'] < 0.5:
            # Low confidence - try causal-focused search
            state['reasoning_chain'].append("Correction: Increasing causal weight")
        else:
            # Moderate confidence - explore more chains
            state['reasoning_chain'].append("Correction: Exploring more graph paths")

        return state

    def _should_correct(self, state: AgentState) -> str:
        """Decide whether to correct or end"""
        if state['verified']:
            return "end"
        if state['confidence'] >= self.min_confidence:
            return "end"
        if state['iteration'] >= state['max_iterations']:
            return "end"
        return "correct"


def create_agent(
    retriever=None,
    graph_engine=None,
    llm_client=None,
    **kwargs
) -> CausalRAGAgent:
    """
    Factory function to create a CausalRAGAgent.

    Args:
        retriever: NeuroCausal retriever
        graph_engine: Graph engine
        llm_client: LLM client
        **kwargs: Additional agent configuration

    Returns:
        Configured CausalRAGAgent instance

    Example:
        >>> from neurocausal_rag.agents import create_agent
        >>> agent = create_agent(retriever, graph, llm)
        >>> result = agent.run("What causes climate change?")
    """
    return CausalRAGAgent(
        retriever=retriever,
        graph_engine=graph_engine,
        llm_client=llm_client,
        **kwargs
    )
