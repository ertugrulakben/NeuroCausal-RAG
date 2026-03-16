import networkx as nx
from copy import deepcopy
from neurocausal_rag.core.graph import GraphEngine

class CausalInferenceEngine:
    def __init__(self, graph_engine: GraphEngine):
        self.graph_engine = graph_engine

    def do_intervention(self, target_node: str, intervention_value: any) -> nx.DiGraph:
        """
        Performs a 'do' operation (intervention) on the graph.
        Removes all incoming edges to the target node and sets its value.
        Returns a new modified graph (mutilated graph).
        """
        # Create a copy of the graph to avoid modifying the original
        mutilated_graph = deepcopy(self.graph_engine.graph)
        
        if not mutilated_graph.has_node(target_node):
            raise ValueError(f"Node {target_node} not found in graph.")
            
        # Remove incoming edges (Pearl's do-calculus rule)
        in_edges = list(mutilated_graph.in_edges(target_node))
        mutilated_graph.remove_edges_from(in_edges)
        
        # Set the intervention value
        mutilated_graph.nodes[target_node]['value'] = intervention_value
        mutilated_graph.nodes[target_node]['intervened'] = True
        
        return mutilated_graph

    def counterfactual_query(self, query_node: str, intervention_node: str, intervention_value: any):
        """
        Answers "What would happen to query_node if intervention_node was set to intervention_value?"
        """
        # 1. Perform intervention
        mutilated_graph = self.do_intervention(intervention_node, intervention_value)
        
        # 2. Propagate effect (Simple simulation)
        # In a real causal model, we would use structural equations.
        # Here we simulate propagation: if A causes B, and A is set, B might change.
        
        # Check if there is a path from intervention_node to query_node
        if nx.has_path(mutilated_graph, intervention_node, query_node):
            # Calculate impact (simplified)
            # Find all paths
            paths = list(nx.all_simple_paths(mutilated_graph, intervention_node, query_node))
            
            return {
                "effect": "possible",
                "paths_count": len(paths),
                "message": f"Intervening on {intervention_node} affects {query_node}."
            }
        else:
            return {
                "effect": "none",
                "message": f"Intervening on {intervention_node} has no causal effect on {query_node}."
            }
