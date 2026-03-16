"""
NeuroCausal RAG - Graph Engine
NetworkX-based causal graph management
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import logging

from .node import NeuroCausalNode
from .edge import NeuroCausalEdge, RelationType, RELATION_TYPE_TO_IDX
from ..config import GraphConfig

logger = logging.getLogger(__name__)


# Weight configuration for relationship types
RELATION_WEIGHTS = {
    RelationType.CAUSES: 1.0,      # Strongest
    RelationType.SUPPORTS: 0.8,    # Medium
    RelationType.REQUIRES: 0.7,    # Medium
    RelationType.RELATED: 0.5      # Weak (RELATED, not RELATED_TO)
}


class GraphEngine:
    """
    NeuroCausal Graph Engine - Knowledge organization and management.

    NetworkX-based graph structure for managing causal relationships.
    Computes node importance using PageRank.
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, NeuroCausalNode] = {}
        self.pagerank_scores: Dict[str, float] = {}

    def add_node(
        self,
        node_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add node to graph"""
        node = NeuroCausalNode(
            id=node_id,
            content=content,
            text_embedding=embedding,
            metadata=metadata or {},
            importance=self.config.importance_default
        )
        node.final_embedding = embedding

        self.nodes[node_id] = node
        self.graph.add_node(
            node_id,
            content=content,
            text_embedding=embedding,
            final_embedding=embedding,
            metadata=metadata or {},
            importance=self.config.importance_default
        )
        self._update_pagerank()
        logger.debug(f"Added node: {node_id}")

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        strength: float = 1.0
    ) -> None:
        """Add edge between nodes"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Source or target node not found: {source} -> {target}")

        rel_type = RelationType(relation_type) if isinstance(relation_type, str) else relation_type

        self.graph.add_edge(
            source,
            target,
            relation_type=rel_type,
            relation_idx=RELATION_TYPE_TO_IDX[rel_type],
            strength=strength
        )
        self._update_pagerank()
        logger.debug(f"Added edge: {source} --{relation_type}--> {target}")

    def _update_pagerank(self) -> None:
        """Update PageRank scores"""
        if len(self.graph.nodes()) > 0:
            try:
                self.pagerank_scores = nx.pagerank(
                    self.graph,
                    weight='strength',
                    alpha=self.config.pagerank_alpha
                )
                # Update node importance
                for node_id, score in self.pagerank_scores.items():
                    if node_id in self.nodes:
                        self.nodes[node_id].importance = score
            except Exception:
                n = len(self.graph.nodes())
                self.pagerank_scores = {node: 1.0/n for node in self.graph.nodes()}

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID"""
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        return {
            'id': node.id,
            'content': node.content,
            'embedding': node.final_embedding,
            'metadata': node.metadata,
            'importance': node.importance
        }

    def get_neighbors(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get neighboring node IDs"""
        if node_id not in self.graph:
            return []

        neighbors = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            if relation_types is None:
                neighbors.append(neighbor)
            elif edge_data['relation_type'].value in relation_types:
                neighbors.append(neighbor)

        return neighbors

    def get_predecessors(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get predecessor node IDs (incoming edges)"""
        if node_id not in self.graph:
            return []

        predecessors = []
        for pred in self.graph.predecessors(node_id):
            edge_data = self.graph.get_edge_data(pred, node_id)
            if relation_types is None:
                predecessors.append(pred)
            elif edge_data['relation_type'].value in relation_types:
                predecessors.append(pred)

        return predecessors

    def get_causal_chain(
        self,
        node_id: str,
        max_depth: int = 3,
        direction: str = 'forward'
    ) -> List[str]:
        """
        Get causal chain starting from a node.
        Follows edges based on RELATION_WEIGHTS (causes, supports, requires, related).

        Args:
            node_id: Starting node
            max_depth: Maximum depth
            direction: 'forward' (to effects) or 'backward' (to causes)
        """
        if node_id not in self.graph:
            return []

        max_depth = min(max_depth, self.config.max_causal_depth)
        chain = [node_id]
        visited = {node_id}
        current = node_id

        for _ in range(max_depth):
            best_next = None
            best_score = 0

            if direction == 'forward':
                # Forward: check outgoing edges
                for neighbor in self.graph.neighbors(current):
                    if neighbor in visited:
                        continue
                    edge_data = self.graph.get_edge_data(current, neighbor)
                    rel_type = edge_data['relation_type']
                    # Use RELATION_WEIGHTS for all relation types
                    weight = RELATION_WEIGHTS.get(rel_type, 0.3)
                    score = weight * edge_data.get('strength', 1.0)
                    if score > best_score:
                        best_next = neighbor
                        best_score = score
            else:
                # Backward: check incoming edges
                for pred in self.graph.predecessors(current):
                    if pred in visited:
                        continue
                    edge_data = self.graph.get_edge_data(pred, current)
                    rel_type = edge_data['relation_type']
                    # Use RELATION_WEIGHTS for all relation types
                    weight = RELATION_WEIGHTS.get(rel_type, 0.3)
                    score = weight * edge_data.get('strength', 1.0)
                    if score > best_score:
                        best_next = pred
                        best_score = score

            if not best_next:
                break

            chain.append(best_next)
            visited.add(best_next)
            current = best_next

        return chain

    def find_causal_path(self, source_id: str, target_id: str) -> Tuple[List[str], float]:
        """Find causal path between two nodes"""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            path_score = 0.0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                path_score += edge_data.get('strength', 1.0)
            path_score = path_score / (len(path) - 1) if len(path) > 1 else 0.0
            return path, path_score
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 0.0

    def get_importance(self, node_id: str) -> float:
        """Get node importance (PageRank)"""
        return self.pagerank_scores.get(node_id, self.config.importance_default)

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all node embeddings and their IDs"""
        ids = list(self.nodes.keys())
        embeddings = np.array([self.nodes[nid].final_embedding for nid in ids])
        return embeddings, ids

    def export(self, path: str) -> None:
        """Export graph to JSON file"""
        data = {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': []
        }

        for source, target, edge_data in self.graph.edges(data=True):
            data['edges'].append({
                'source': source,
                'target': target,
                'relation_type': edge_data['relation_type'].value,
                'strength': edge_data['strength']
            })

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph exported to {path}")

    def load(self, path: str) -> None:
        """Load graph from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.graph.clear()
        self.nodes.clear()

        for node_data in data['nodes']:
            node = NeuroCausalNode.from_dict(node_data)
            self.nodes[node.id] = node
            self.graph.add_node(
                node.id,
                content=node.content,
                text_embedding=node.text_embedding,
                final_embedding=node.final_embedding,
                metadata=node.metadata,
                importance=node.importance
            )

        for edge_data in data['edges']:
            edge = NeuroCausalEdge.from_dict(edge_data)
            self.graph.add_edge(
                edge.source,
                edge.target,
                relation_type=edge.relation_type,
                relation_idx=RELATION_TYPE_TO_IDX[edge.relation_type],
                strength=edge.strength
            )

        self._update_pagerank()
        logger.info(f"Graph loaded from {path}: {len(self.nodes)} nodes, {self.graph.number_of_edges()} edges")

    @property
    def node_count(self) -> int:
        """Return number of nodes"""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return number of edges"""
        return self.graph.number_of_edges()

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0,
            'is_connected': nx.is_weakly_connected(self.graph) if self.nodes else False,
            'num_relation_types': len(set(d['relation_type'] for _, _, d in self.graph.edges(data=True)))
        }


class Neo4jGraphEngine:
    """
    Neo4j-based Graph Engine for production scale.

    Implements the same interface as GraphEngine but uses Neo4j
    as the backend for enterprise-grade graph storage and querying.
    """

    def __init__(self, config: Optional[GraphConfig] = None, neo4j_config=None):
        from ..config import Neo4jConfig

        self.config = config or GraphConfig()
        self.neo4j_config = neo4j_config or Neo4jConfig()

        self._driver = None
        self._pagerank_scores: Dict[str, float] = {}
        self._nodes_cache: Dict[str, NeuroCausalNode] = {}

    def _get_driver(self):
        """Lazy initialization of Neo4j driver"""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.neo4j_config.uri,
                    auth=(self.neo4j_config.user, self.neo4j_config.password),
                    max_connection_pool_size=self.neo4j_config.max_connection_pool_size,
                    connection_timeout=self.neo4j_config.connection_timeout
                )
                logger.info(f"Connected to Neo4j at {self.neo4j_config.uri}")
            except ImportError:
                raise ImportError("neo4j package required. Install with: pip install neo4j>=5.0")
        return self._driver

    def close(self):
        """Close Neo4j connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def add_node(
        self,
        node_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add node to Neo4j graph"""
        driver = self._get_driver()

        # Convert embedding to list for storage
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        meta = metadata or {}

        query = """
        MERGE (n:Document {id: $node_id})
        SET n.content = $content,
            n.embedding = $embedding,
            n.metadata = $metadata,
            n.importance = $importance,
            n.updated_at = datetime()
        RETURN n.id
        """

        with driver.session(database=self.neo4j_config.database) as session:
            session.run(
                query,
                node_id=node_id,
                content=content,
                embedding=embedding_list,
                metadata=json.dumps(meta),
                importance=self.config.importance_default
            )

        # Cache the node
        node = NeuroCausalNode(
            id=node_id,
            content=content,
            text_embedding=embedding,
            metadata=meta,
            importance=self.config.importance_default
        )
        node.final_embedding = embedding
        self._nodes_cache[node_id] = node

        logger.debug(f"Added node to Neo4j: {node_id}")

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        strength: float = 1.0
    ) -> None:
        """Add edge between nodes in Neo4j"""
        driver = self._get_driver()

        rel_type = RelationType(relation_type) if isinstance(relation_type, str) else relation_type
        rel_name = rel_type.value.upper()  # Neo4j convention: uppercase relation names

        query = f"""
        MATCH (s:Document {{id: $source}})
        MATCH (t:Document {{id: $target}})
        MERGE (s)-[r:{rel_name}]->(t)
        SET r.strength = $strength,
            r.relation_idx = $relation_idx,
            r.created_at = datetime()
        RETURN type(r)
        """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(
                query,
                source=source,
                target=target,
                strength=strength,
                relation_idx=RELATION_TYPE_TO_IDX[rel_type]
            )
            if not result.single():
                raise ValueError(f"Failed to create edge: {source} -> {target}")

        logger.debug(f"Added edge to Neo4j: {source} --{relation_type}--> {target}")

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID from Neo4j"""
        # Check cache first
        if node_id in self._nodes_cache:
            node = self._nodes_cache[node_id]
            return {
                'id': node.id,
                'content': node.content,
                'embedding': node.final_embedding,
                'metadata': node.metadata,
                'importance': node.importance
            }

        driver = self._get_driver()

        query = """
        MATCH (n:Document {id: $node_id})
        RETURN n.id as id, n.content as content, n.embedding as embedding,
               n.metadata as metadata, n.importance as importance
        """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query, node_id=node_id)
            record = result.single()

            if not record:
                return None

            metadata = json.loads(record['metadata']) if record['metadata'] else {}
            embedding = np.array(record['embedding']) if record['embedding'] else None

            return {
                'id': record['id'],
                'content': record['content'],
                'embedding': embedding,
                'metadata': metadata,
                'importance': record['importance'] or self.config.importance_default
            }

    def get_neighbors(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get neighboring node IDs (outgoing edges)"""
        driver = self._get_driver()

        if relation_types:
            rel_filter = '|'.join([r.upper() for r in relation_types])
            query = f"""
            MATCH (n:Document {{id: $node_id}})-[r:{rel_filter}]->(m:Document)
            RETURN m.id as neighbor_id
            """
        else:
            query = """
            MATCH (n:Document {id: $node_id})-[r]->(m:Document)
            RETURN m.id as neighbor_id
            """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query, node_id=node_id)
            return [record['neighbor_id'] for record in result]

    def get_predecessors(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get predecessor node IDs (incoming edges)"""
        driver = self._get_driver()

        if relation_types:
            rel_filter = '|'.join([r.upper() for r in relation_types])
            query = f"""
            MATCH (m:Document)-[r:{rel_filter}]->(n:Document {{id: $node_id}})
            RETURN m.id as pred_id
            """
        else:
            query = """
            MATCH (m:Document)-[r]->(n:Document {id: $node_id})
            RETURN m.id as pred_id
            """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query, node_id=node_id)
            return [record['pred_id'] for record in result]

    def get_causal_chain(
        self,
        node_id: str,
        max_depth: int = 3,
        direction: str = 'forward'
    ) -> List[str]:
        """Get causal chain using Cypher path queries"""
        driver = self._get_driver()
        max_depth = min(max_depth, self.config.max_causal_depth)

        # Build relationship pattern with weights
        rel_types = "CAUSES|SUPPORTS|REQUIRES|RELATED"

        if direction == 'forward':
            query = f"""
            MATCH path = (start:Document {{id: $node_id}})-[r:{rel_types}*1..{max_depth}]->(end:Document)
            WITH path,
                 reduce(score = 0.0, rel in relationships(path) |
                   score + CASE type(rel)
                     WHEN 'CAUSES' THEN 1.0
                     WHEN 'SUPPORTS' THEN 0.8
                     WHEN 'REQUIRES' THEN 0.7
                     WHEN 'RELATED' THEN 0.5
                     ELSE 0.3
                   END * coalesce(rel.strength, 1.0)
                 ) as path_score
            ORDER BY path_score DESC
            LIMIT 1
            RETURN [n in nodes(path) | n.id] as chain
            """
        else:
            query = f"""
            MATCH path = (start:Document)-[r:{rel_types}*1..{max_depth}]->(end:Document {{id: $node_id}})
            WITH path,
                 reduce(score = 0.0, rel in relationships(path) |
                   score + CASE type(rel)
                     WHEN 'CAUSES' THEN 1.0
                     WHEN 'SUPPORTS' THEN 0.8
                     WHEN 'REQUIRES' THEN 0.7
                     WHEN 'RELATED' THEN 0.5
                     ELSE 0.3
                   END * coalesce(rel.strength, 1.0)
                 ) as path_score
            ORDER BY path_score DESC
            LIMIT 1
            RETURN [n in nodes(path) | n.id] as chain
            """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query, node_id=node_id)
            record = result.single()

            if not record or not record['chain']:
                return [node_id]

            chain = record['chain']
            # For backward direction, the chain is already in correct order (cause -> effect)
            return chain

    def find_causal_path(self, source_id: str, target_id: str) -> Tuple[List[str], float]:
        """Find shortest path between two nodes"""
        driver = self._get_driver()

        query = """
        MATCH path = shortestPath((s:Document {id: $source})-[*]-(t:Document {id: $target}))
        WITH path,
             reduce(score = 0.0, rel in relationships(path) |
               score + coalesce(rel.strength, 1.0)
             ) as total_score
        RETURN [n in nodes(path) | n.id] as path_nodes,
               total_score / size(relationships(path)) as avg_score
        """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query, source=source_id, target=target_id)
            record = result.single()

            if not record:
                return [], 0.0

            return record['path_nodes'], record['avg_score']

    def get_importance(self, node_id: str) -> float:
        """Get node importance (cached PageRank or compute)"""
        if node_id in self._pagerank_scores:
            return self._pagerank_scores[node_id]
        return self.config.importance_default

    def update_pagerank(self) -> None:
        """
        Compute PageRank using Neo4j GDS (Graph Data Science) if available,
        otherwise use a simple degree-based approximation.
        """
        driver = self._get_driver()

        # Try GDS PageRank first
        try:
            query = """
            CALL gds.pageRank.stream('document-graph', {
                dampingFactor: $alpha,
                relationshipWeightProperty: 'strength'
            })
            YIELD nodeId, score
            MATCH (n) WHERE id(n) = nodeId
            RETURN n.id as node_id, score
            """

            with driver.session(database=self.neo4j_config.database) as session:
                result = session.run(query, alpha=self.config.pagerank_alpha)
                for record in result:
                    self._pagerank_scores[record['node_id']] = record['score']
                logger.info("PageRank computed using GDS")
                return
        except Exception as e:
            logger.debug(f"GDS PageRank not available: {e}")

        # Fallback: degree-based importance
        query = """
        MATCH (n:Document)
        OPTIONAL MATCH (n)-[r]->()
        WITH n, count(r) as out_degree
        OPTIONAL MATCH ()-[r]->(n)
        WITH n, out_degree, count(r) as in_degree
        RETURN n.id as node_id,
               toFloat(out_degree + in_degree * 2) / 10.0 as score
        """

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query)
            total = 0.0
            scores = {}
            for record in result:
                scores[record['node_id']] = record['score']
                total += record['score']

            # Normalize
            if total > 0:
                for node_id, score in scores.items():
                    self._pagerank_scores[node_id] = score / total

        logger.info(f"PageRank approximated for {len(self._pagerank_scores)} nodes")

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all node embeddings and their IDs"""
        driver = self._get_driver()

        query = """
        MATCH (n:Document)
        WHERE n.embedding IS NOT NULL
        RETURN n.id as id, n.embedding as embedding
        ORDER BY n.id
        """

        ids = []
        embeddings = []

        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run(query)
            for record in result:
                ids.append(record['id'])
                embeddings.append(record['embedding'])

        if not embeddings:
            return np.array([]), []

        return np.array(embeddings), ids

    def export(self, path: str) -> None:
        """Export graph to JSON file"""
        driver = self._get_driver()

        data = {'nodes': [], 'edges': []}

        # Export nodes
        with driver.session(database=self.neo4j_config.database) as session:
            nodes_result = session.run("""
                MATCH (n:Document)
                RETURN n.id as id, n.content as content, n.embedding as embedding,
                       n.metadata as metadata, n.importance as importance
            """)
            for record in nodes_result:
                data['nodes'].append({
                    'id': record['id'],
                    'content': record['content'],
                    'embedding': record['embedding'],
                    'metadata': json.loads(record['metadata']) if record['metadata'] else {},
                    'importance': record['importance']
                })

            # Export edges
            edges_result = session.run("""
                MATCH (s:Document)-[r]->(t:Document)
                RETURN s.id as source, t.id as target,
                       toLower(type(r)) as relation_type, r.strength as strength
            """)
            for record in edges_result:
                data['edges'].append({
                    'source': record['source'],
                    'target': record['target'],
                    'relation_type': record['relation_type'],
                    'strength': record['strength'] or 1.0
                })

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Neo4j graph exported to {path}")

    def load(self, path: str) -> None:
        """Load graph from JSON file into Neo4j"""
        driver = self._get_driver()

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Clear existing data
        with driver.session(database=self.neo4j_config.database) as session:
            session.run("MATCH (n:Document) DETACH DELETE n")

        # Load nodes
        for node_data in data['nodes']:
            self.add_node(
                node_id=node_data['id'],
                content=node_data['content'],
                embedding=np.array(node_data['embedding']),
                metadata=node_data.get('metadata', {})
            )

        # Load edges
        for edge_data in data['edges']:
            self.add_edge(
                source=edge_data['source'],
                target=edge_data['target'],
                relation_type=edge_data['relation_type'],
                strength=edge_data.get('strength', 1.0)
            )

        self.update_pagerank()
        logger.info(f"Graph loaded to Neo4j from {path}")

    @property
    def node_count(self) -> int:
        """Return number of nodes"""
        driver = self._get_driver()
        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run("MATCH (n:Document) RETURN count(n) as count")
            return result.single()['count']

    @property
    def edge_count(self) -> int:
        """Return number of edges"""
        driver = self._get_driver()
        with driver.session(database=self.neo4j_config.database) as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            return result.single()['count']

    # Compatibility property for retriever
    @property
    def nodes(self) -> Dict[str, NeuroCausalNode]:
        """Return cached nodes dict for compatibility"""
        return self._nodes_cache

    @property
    def graph(self):
        """Return self for compatibility with NetworkX-based code"""
        return self

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics from Neo4j"""
        driver = self._get_driver()

        with driver.session(database=self.neo4j_config.database) as session:
            stats_result = session.run("""
                MATCH (n:Document)
                OPTIONAL MATCH (n)-[r]->()
                WITH count(DISTINCT n) as total_nodes,
                     count(r) as total_edges,
                     avg(count(r)) as avg_degree
                RETURN total_nodes, total_edges, avg_degree
            """)
            stats = stats_result.single()

            # Check connectivity
            conn_result = session.run("""
                CALL gds.wcc.stream('document-graph')
                YIELD componentId
                RETURN count(DISTINCT componentId) as components
            """)

            try:
                conn = conn_result.single()
                is_connected = conn['components'] == 1
            except Exception:
                is_connected = False

            # Count relation types
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN count(DISTINCT type(r)) as rel_types
            """)
            rel_count = rel_result.single()['rel_types']

            return {
                'total_nodes': stats['total_nodes'],
                'total_edges': stats['total_edges'],
                'avg_degree': stats['avg_degree'] or 0,
                'is_connected': is_connected,
                'num_relation_types': rel_count
            }


def create_graph_engine(config: Optional[GraphConfig] = None, neo4j_config=None):
    """
    Factory function to create appropriate graph engine.

    Returns Neo4jGraphEngine if backend is 'neo4j', otherwise GraphEngine (NetworkX).
    """
    config = config or GraphConfig()

    if config.backend == "neo4j":
        return Neo4jGraphEngine(config=config, neo4j_config=neo4j_config)
    else:
        return GraphEngine(config=config)