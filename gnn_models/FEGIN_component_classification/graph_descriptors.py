import torch
import numpy as np
import networkx as nx
import dgsd
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


def compute_netlsd(G, n_eigenvalues=10):
    # NetLSD (Network Laplacian Spectral Descriptor).
    # global graph-level feature based on the graph's spectral properties (trace of heat kernel matrix at diff time scales)
    
    if G.number_of_nodes() == 0:
        return np.zeros(n_eigenvalues)
    
    if G.number_of_nodes() == 1:
        return np.zeros(n_eigenvalues)
    
    try:
        # normalized Laplacian
        L = nx.normalized_laplacian_matrix(G)
        
        # Eigenvalues
        n_nodes = G.number_of_nodes()
        k = min(n_eigenvalues, n_nodes - 2)
        
        if k < 1:
            return np.zeros(n_eigenvalues)
        
        # k smallest eigenvalues
        eigenvalues = sp_linalg.eigsh(L, k=k, which='SM', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        
        # padding
        if len(eigenvalues) < n_eigenvalues:
            eigenvalues = np.pad(eigenvalues, (0, n_eigenvalues - len(eigenvalues)), 
                                mode='constant', constant_values=0)
        
        return eigenvalues[:n_eigenvalues]
        
    except Exception as e:
        print(f"Warning: NetLSD computation failed: {e}")
        return np.zeros(n_eigenvalues)
    
def compute_dgsd(G, bins=10):
    # Distributional Graph Structure Descriptor
    # Histogram of shortest path lengths between all node pairs
    return dgsd.DGSD().get_descriptor(G, bins=bins)


def compute_basic_stats(G):

    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0,
            'avg_degree': 0,
            'avg_clustering': 0
        }
    
    # Density
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0
    
    # Average degree
    avg_degree = 2 * m / n if n > 0 else 0
    
    # Average clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0
    
    return {
        'num_nodes': n,
        'num_edges': m,
        'density': density,
        'avg_degree': avg_degree,
        'avg_clustering': avg_clustering
    }


def compute_degree_stats(G):

    if G.number_of_nodes() == 0:
        return {
            'degree_mean': 0,
            'degree_std': 0,
            'degree_max': 0,
            'degree_min': 0
        }
    
    degrees = [d for n, d in G.degree()]
    
    return {
        'degree_mean': np.mean(degrees),
        'degree_std': np.std(degrees),
        'degree_max': np.max(degrees),
        'degree_min': np.min(degrees)
    }


def compute_component_type_distribution(G):

    from collections import Counter
    
    COMPONENT_TYPES = ["R", "C", "L", "V", "M", "Q", "D", "I"]
    
    comp_types = []
    for node, attr in G.nodes(data=True):
        if attr.get("type") == "component":
            comp_type = attr.get("comp_type")
            if comp_type in COMPONENT_TYPES:
                comp_types.append(comp_type)
    
    counter = Counter(comp_types)
    total = len(comp_types) if comp_types else 1
    
    # Return normalized distribution
    distribution = []
    for ct in COMPONENT_TYPES:
        distribution.append(counter[ct] / total)
    
    return np.array(distribution)


def compute_all_graph_descriptors(G, n_eigenvalues=10):
    # torch.Tensor: Graph descriptor vector

    descriptors = []
    
    # 1. NetLSD (spectral descriptor)
    netlsd = compute_netlsd(G, n_eigenvalues)
    descriptors.extend(netlsd)

    # 2. DGSD (pairwise distance descriptor)
    dgsd = compute_dgsd(G, n_eigenvalues)
    descriptors.extend(dgsd)
    
    # 3. Basic statistics
    basic = compute_basic_stats(G)
    descriptors.extend([
        basic['num_nodes'],
        basic['num_edges'],
        basic['density'],
        basic['avg_degree'],
        basic['avg_clustering']
    ])
    
    # 4. Degree statistics
    degree_stats = compute_degree_stats(G)
    descriptors.extend([
        degree_stats['degree_mean'],
        degree_stats['degree_std'],
        degree_stats['degree_max'],
        degree_stats['degree_min']
    ])
    
    # 5. Component type distribution
    comp_dist = compute_component_type_distribution(G)
    descriptors.extend(comp_dist)
    
    return torch.tensor(descriptors, dtype=torch.float32)


def get_descriptor_dimension(n_eigenvalues=10, dgsd_bins=10):
    # NetLSD & DGSD (n_eigenvalues, dgsd_bins) + basic stats (5) + degree stats (4) + comp distribution (8)
    return n_eigenvalues + dgsd_bins + 5 + 4 + 8


class GraphDescriptorCache:
    # cache computed graph descriptors to avoid recomputation
    def __init__(self, n_eigenvalues=10, dgsd_bins=10):
        self.n_eigenvalues = n_eigenvalues
        self.dgsd_bins = dgsd_bins
        self.cache = {}
    
    def get_or_compute(self, graph_id, G):
        if graph_id not in self.cache:
            self.cache[graph_id] = compute_all_graph_descriptors(G, self.n_eigenvalues)
        return self.cache[graph_id]
    
    def clear(self):
        self.cache.clear()


if __name__ == "__main__":
    # Test graph descriptor computation
    print("Testing Graph Descriptor Computation...\n")
    
    # simple test graph
    G = nx.Graph()
    G.add_nodes_from([
        ('R1', {'type': 'component', 'comp_type': 'R'}),
        ('R2', {'type': 'component', 'comp_type': 'R'}),
        ('C1', {'type': 'component', 'comp_type': 'C'}),
        ('V1', {'type': 'component', 'comp_type': 'V'}),
    ])
    G.add_edges_from([('R1', 'C1'), ('C1', 'R2'), ('R2', 'V1')])
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # descriptors
    descriptors = compute_all_graph_descriptors(G)
    
    print(f"Descriptor dimension: {len(descriptors)}")
    print(f"Expected dimension: {get_descriptor_dimension()}")
    print(f"Descriptor values (first 20):")
    print(descriptors[:20])
    
    # Individual components
    print("Individual Descriptor Components:")
    
    netlsd = compute_netlsd(G)
    print(f"NetLSD ({len(netlsd)} dims): {netlsd}")

    dgsd = compute_dgsd(G)
    print(f"DGSD ({len(dgsd)} dims): {dgsd}")
    
    basic = compute_basic_stats(G)
    print(f"Basic stats:")
    for k, v in basic.items():
        print(f"  {k}: {v:.4f}")
    
    degree_stats = compute_degree_stats(G)
    print(f"Degree stats:")
    for k, v in degree_stats.items():
        print(f"  {k}: {v:.4f}")
    
    comp_dist = compute_component_type_distribution(G)
    print(f"Component distribution:")
    COMPONENT_TYPES = ["R", "C", "L", "V", "M", "Q", "D", "I"]
    for ct, val in zip(COMPONENT_TYPES, comp_dist):
        if val > 0:
            print(f"  {ct}: {val:.4f}")
    
    print("Graph descriptors ready for FEGIN")
