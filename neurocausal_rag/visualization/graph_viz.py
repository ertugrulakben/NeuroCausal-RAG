"""
NeuroCausal RAG - Graph Visualization
PyVis ile interaktif graf gorsellestirme

Author: Ertugrul Akben
"""

from typing import List, Dict, Optional, Any
from pyvis.network import Network
import tempfile
import os

from ..core.edge import RelationType


# Relation type colors
RELATION_COLORS = {
    RelationType.CAUSES: "#ff6b6b",      # Red - causality
    RelationType.SUPPORTS: "#4ecdc4",    # Turquoise - support
    RelationType.REQUIRES: "#ffe66d",    # Yellow - requirement
    RelationType.RELATED: "#95afc0",     # Gray - relation
    RelationType.CONTRADICTS: "#eb4d4b", # Dark red - contradiction
}

# Relation type widths (strength * multiplier)
RELATION_WIDTHS = {
    RelationType.CAUSES: 4,
    RelationType.SUPPORTS: 3,
    RelationType.REQUIRES: 2,
    RelationType.RELATED: 1,
    RelationType.CONTRADICTS: 2,
}


class CausalGraphVisualizer:
    """
    Causal graph visualizer.

    Creates interactive HTML graphs using PyVis.
    """

    def __init__(self, graph_engine, height: str = "600px", width: str = "100%"):
        """
        Args:
            graph_engine: GraphEngine instance
            height: Graph height
            width: Graph width
        """
        self.graph = graph_engine
        self.height = height
        self.width = width

    def _create_network(self, physics: bool = True) -> Network:
        """Create new PyVis Network"""
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor="#0e1117",  # Streamlit dark theme
            font_color="#fafafa",
            directed=True,
            notebook=False,
            cdn_resources='remote'
        )

        # Physics settings
        if physics:
            net.set_options("""
            {
                "nodes": {
                    "font": {"size": 14, "face": "Arial"},
                    "borderWidth": 2,
                    "shadow": true
                },
                "edges": {
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                    "smooth": {"type": "curvedCW", "roundness": 0.2},
                    "shadow": true
                },
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.05,
                        "damping": 0.5
                    }
                },
                "interaction": {
                    "hover": true,
                    "navigationButtons": true,
                    "keyboard": true
                }
            }
            """)
        else:
            net.set_options("""
            {
                "physics": {"enabled": false},
                "interaction": {"hover": true, "navigationButtons": true}
            }
            """)

        return net

    def _truncate_content(self, content: str, max_len: int = 50) -> str:
        """Truncate content"""
        if len(content) <= max_len:
            return content
        return content[:max_len] + "..."

    def _get_node_title(self, node_id: str) -> str:
        """Create node tooltip content"""
        node = self.graph.get_node(node_id)
        if not node:
            return node_id

        importance = node.get('importance', 0)
        content = node.get('content', '')[:200]
        metadata = node.get('metadata', {})
        category = metadata.get('category', 'Unknown')

        return f"""<b>{node_id}</b>
<hr>
<b>Kategori:</b> {category}<br>
<b>Onem:</b> {importance:.4f}<br>
<hr>
{content}..."""

    def visualize_full_graph(self, max_nodes: int = 100) -> str:
        """
        Visualize full graph.

        Args:
            max_nodes: Maximum number of nodes to display

        Returns:
            HTML string
        """
        net = self._create_network()

        # Add nodes
        nodes_added = 0
        for node_id in self.graph.nodes:
            if nodes_added >= max_nodes:
                break

            node = self.graph.get_node(node_id)
            if not node:
                continue

            importance = node.get('importance', 0)
            # Size: based on importance (min 20, max 50)
            size = 20 + (importance * 300)
            size = max(20, min(50, size))

            net.add_node(
                node_id,
                label=self._truncate_content(node_id, 20),
                title=self._get_node_title(node_id),
                size=size,
                color="#6c5ce7",  # Mor - normal node
                shape="dot"
            )
            nodes_added += 1

        # Add edges
        for source, target, data in self.graph.graph.edges(data=True):
            if source not in [n['id'] for n in net.nodes] or target not in [n['id'] for n in net.nodes]:
                continue

            rel_type = data.get('relation_type', RelationType.RELATED)
            strength = data.get('strength', 1.0)

            color = RELATION_COLORS.get(rel_type, "#95afc0")
            width = RELATION_WIDTHS.get(rel_type, 1) * strength

            net.add_edge(
                source,
                target,
                title=f"{rel_type.value} ({strength:.2f})",
                color=color,
                width=width
            )

        # Generate HTML
        return self._generate_html(net)

    def visualize_search_results(
        self,
        query: str,
        results: List[Any],
        chains: Optional[Dict[str, List[str]]] = None,
        show_all_connections: bool = True
    ) -> str:
        """
        Visualize search results.

        Args:
            query: Search query
            results: SearchResult listesi
            chains: Causal chain for each result {node_id: [chain_nodes]}
            show_all_connections: Show all connections

        Returns:
            HTML string
        """
        net = self._create_network()

        # Query node (red)
        net.add_node(
            "QUERY",
            label=self._truncate_content(query, 30),
            title=f"<b>Sorgu:</b><br>{query}",
            size=45,
            color="#e74c3c",  # Kirmizi
            shape="star",
            font={"size": 16, "color": "#ffffff"}
        )

        # Collect result nodes and chains
        all_nodes = set()
        result_nodes = set()
        injected_nodes = set()

        for i, result in enumerate(results):
            node_id = result.node_id if hasattr(result, 'node_id') else result.get('node_id', str(i))
            result_nodes.add(node_id)
            all_nodes.add(node_id)

            # Check if injected
            metadata = result.metadata if hasattr(result, 'metadata') else result.get('metadata', {})
            if metadata.get('injected_from'):
                injected_nodes.add(node_id)

            # Add chain nodes
            if chains and node_id in chains:
                for chain_node in chains[node_id]:
                    all_nodes.add(chain_node)

        # Add nodes
        for node_id in all_nodes:
            node = self.graph.get_node(node_id)
            if not node:
                continue

            importance = node.get('importance', 0)

            # Determine color and size
            if node_id in result_nodes:
                if node_id in injected_nodes:
                    color = "#f39c12"  # Turuncu - enjekte
                    size = 35
                else:
                    color = "#27ae60"  # Yesil - sonuc
                    size = 40
            else:
                color = "#3498db"  # Mavi - zincir
                size = 25

            net.add_node(
                node_id,
                label=self._truncate_content(node_id, 20),
                title=self._get_node_title(node_id),
                size=size,
                color=color,
                shape="dot"
            )

        # Query -> Result edges
        for i, result in enumerate(results):
            node_id = result.node_id if hasattr(result, 'node_id') else result.get('node_id', str(i))
            score = result.score if hasattr(result, 'score') else result.get('score', 0)

            net.add_edge(
                "QUERY",
                node_id,
                title=f"Skor: {score:.3f}",
                color="#e74c3c",
                width=2 + (score * 3),
                dashes=True
            )

        # Add actual graph edges
        if show_all_connections:
            added_edges = set()
            for source, target, data in self.graph.graph.edges(data=True):
                if source in all_nodes and target in all_nodes:
                    edge_key = (source, target)
                    if edge_key in added_edges:
                        continue
                    added_edges.add(edge_key)

                    rel_type = data.get('relation_type', RelationType.RELATED)
                    strength = data.get('strength', 1.0)

                    color = RELATION_COLORS.get(rel_type, "#95afc0")
                    width = RELATION_WIDTHS.get(rel_type, 1) * strength

                    net.add_edge(
                        source,
                        target,
                        title=f"{rel_type.value} ({strength:.2f})",
                        color=color,
                        width=width
                    )

        return self._generate_html(net)

    def visualize_causal_chain(
        self,
        source_id: str,
        target_id: Optional[str] = None,
        max_depth: int = 4
    ) -> str:
        """
        Visualize causal chain.

        Args:
            source_id: Start node
            target_id: End node (optional)
            max_depth: Maximum depth

        Returns:
            HTML string
        """
        net = self._create_network(physics=False)

        if target_id:
            # Path between two nodes
            path, score = self.graph.find_causal_path(source_id, target_id)
            chain = path
        else:
            # Forward chain
            chain = self.graph.get_causal_chain(source_id, max_depth, 'forward')

        if not chain:
            # Empty graph
            net.add_node(
                "empty",
                label="No chain found",
                color="#e74c3c",
                size=30
            )
            return self._generate_html(net)

        # Add nodes sequentially (left to right)
        x_pos = 0
        for i, node_id in enumerate(chain):
            node = self.graph.get_node(node_id)

            # First node green, last node red, middle ones blue
            if i == 0:
                color = "#27ae60"  # Yesil - baslangic
                size = 40
            elif i == len(chain) - 1:
                color = "#e74c3c"  # Kirmizi - bitis
                size = 40
            else:
                color = "#3498db"  # Mavi - ara
                size = 30

            net.add_node(
                node_id,
                label=self._truncate_content(node_id, 25),
                title=self._get_node_title(node_id) if node else node_id,
                size=size,
                color=color,
                x=x_pos,
                y=0,
                physics=False
            )
            x_pos += 200

        # Add chain edges
        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]

            # Get actual edge data
            edge_data = self.graph.graph.get_edge_data(source, target)
            if edge_data:
                rel_type = edge_data.get('relation_type', RelationType.CAUSES)
                strength = edge_data.get('strength', 1.0)
            else:
                rel_type = RelationType.CAUSES
                strength = 1.0

            color = RELATION_COLORS.get(rel_type, "#ff6b6b")

            net.add_edge(
                source,
                target,
                title=f"{rel_type.value} ({strength:.2f})",
                color=color,
                width=4,
                label=rel_type.value
            )

        return self._generate_html(net)

    def _generate_html(self, net: Network) -> str:
        """
        Convert PyVis Network to HTML string.

        Creates a temporary file, reads it, then deletes it.
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            temp_path = f.name

        try:
            # Write HTML to file
            net.save_graph(temp_path)

            # Read file
            with open(temp_path, 'r', encoding='utf-8') as f:
                html = f.read()

            return html
        finally:
            # Delete temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_legend_html(self) -> str:
        """Create graph legend HTML"""
        return """
        <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h4 style="color: #fafafa; margin-bottom: 10px;">Renk Kodlari</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                <span style="color: #e74c3c;">&#9679; Sorgu / Bitis</span>
                <span style="color: #27ae60;">&#9679; Arama Sonucu / Baslangic</span>
                <span style="color: #f39c12;">&#9679; Enjekte Edilmis</span>
                <span style="color: #3498db;">&#9679; Zincir Node</span>
                <span style="color: #6c5ce7;">&#9679; Normal Node</span>
            </div>
            <h4 style="color: #fafafa; margin-top: 15px; margin-bottom: 10px;">Iliski Tipleri</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                <span style="color: #ff6b6b;">&#8594; causes (1.0)</span>
                <span style="color: #4ecdc4;">&#8594; supports (0.8)</span>
                <span style="color: #ffe66d;">&#8594; requires (0.7)</span>
                <span style="color: #95afc0;">&#8594; related (0.5)</span>
            </div>
        </div>
        """


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_graph_visualization(
    graph_engine,
    result_ids: List[str] = None,
    height: str = "500px"
) -> str:
    """
    Helper function for graph visualization.

    Args:
        graph_engine: GraphEngine instance
        result_ids: Node IDs to highlight
        height: Graph height

    Returns:
        HTML string
    """
    visualizer = CausalGraphVisualizer(graph_engine, height=height)

    if result_ids:
        # Highlight results (simple mode)
        return visualizer.visualize_full_graph(max_nodes=50)
    else:
        return visualizer.visualize_full_graph(max_nodes=50)
