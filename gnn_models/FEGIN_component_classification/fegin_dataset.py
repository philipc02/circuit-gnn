# FIXED Dataset loader for FEGIN component classification
# Key fix: Component representation now keeps graph connectivity

import torch
from torch_geometric.data import Data, Dataset, Batch
import networkx as nx
import pickle
import os
from graph_descriptors import GraphDescriptorCache
import numpy as np

COMPONENT_TYPES = ["R", "C", "V", "X"]


class FEGINDatasetFiltered(Dataset):
    """
    FIXED: Component representation now maintains graph connectivity
    by keeping the full star graph but zeroing out features appropriately
    """
    
    def __init__(self, graph_files, representation='star', 
                 mask_strategy='keep_pins', n_eigenvalues=10, dgsd_bins=10, 
                 masks_per_graph=4, training=True):
        super().__init__()
        
        self.graph_files = graph_files
        self.representation = representation
        self.mask_strategy = mask_strategy
        self.n_eigenvalues = n_eigenvalues
        self.dgsd_bins = dgsd_bins
        self.masks_per_graph = masks_per_graph
        self.training = training
        
        self.graph_data_cache = {}
        self.create_masks()
        self.descriptor_cache = GraphDescriptorCache(n_eigenvalues, dgsd_bins)
    
    def create_masks(self):
        for folder, filename in self.graph_files:
            graph_path = os.path.join(folder, filename)
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            
            # Find maskable components (R, C, V) and subcircuits (X)
            component_nodes = [n for n, attr in G.nodes(data=True) 
                              if (attr.get("type") == "component" and attr.get("comp_type") in ["R", "C", "V"])
                              or (attr.get("type") == "subcircuit" and attr.get("comp_type") == "X")]
            
            self.graph_data_cache[(folder, filename)] = {
                'original_graph': G,
                'maskable_components': component_nodes
            }
    
    def len(self):
        return len(self.graph_files) * self.masks_per_graph
    
    def get(self, idx):
        graph_idx = idx // self.masks_per_graph
        mask_idx = idx % self.masks_per_graph

        folder, filename = self.graph_files[graph_idx]
        cache_data = self.graph_data_cache[(folder, filename)]
        G = cache_data['original_graph']
        maskable_components = cache_data['maskable_components']
        
        if len(maskable_components) == 0:
            return None
        
        # Deterministic masking based on idx
        rng = np.random.RandomState(seed=idx)
        
        # Select component to mask
        if mask_idx < len(maskable_components):
            masked_component = maskable_components[mask_idx]
        else:
            masked_component = maskable_components[mask_idx % len(maskable_components)]
        
        comp_type = G.nodes[masked_component].get("comp_type")
        
        if comp_type not in COMPONENT_TYPES:
            return None
        
        # Create masked graph (keeps full structure!)
        G_masked = self.create_masked_graph(G, masked_component)
        
        # Data augmentation for training
        if self.training and rng.random() < 0.5:
            G_masked = self.augment_training_data(G_masked, masked_component, comp_type, rng)
        
        # Convert to PyG Data
        data = self.graph_to_data(G_masked)
        
        # Add graph descriptors
        descriptor_id = f"{filename}_mask{mask_idx}"
        descriptor = self.descriptor_cache.get_or_compute(descriptor_id, G_masked)
        data.graph_descriptor = descriptor
        
        # Add label
        data.y = torch.tensor(COMPONENT_TYPES.index(comp_type), dtype=torch.long)
        data.masked_component = masked_component
        data.graph_id = f"{filename}_mask{mask_idx}"
        
        return data
    
    def create_masked_graph(self, G, masked_component):
        """Creates masked version of graph by setting mask tokens"""
        G_masked = G.copy()
        
        # Mark component as masked
        G_masked.nodes[masked_component]['is_masked'] = True
        
        # Set mask token for component features
        if 'features' in G_masked.nodes[masked_component]:
            original_comp_type = G_masked.nodes[masked_component]['features'].get('comp_type_idx', -1)
            G_masked.nodes[masked_component]['features']['original_comp_type'] = original_comp_type
            G_masked.nodes[masked_component]['features']['comp_type_idx'] = 4  # mask token
        
        # Mask associated pins
        for neighbor in G.neighbors(masked_component):
            node_attr = G.nodes[neighbor]
            if (node_attr.get("type") == "pin" and 
                node_attr.get("component") == masked_component):
                G_masked.nodes[neighbor]['is_masked'] = True
                if 'features' in G_masked.nodes[neighbor]:
                    original_pin_type = G_masked.nodes[neighbor]['features'].get('pin_type_idx', -1)
                    G_masked.nodes[neighbor]['features']['original_pin_type'] = original_pin_type
                    G_masked.nodes[neighbor]['features']['pin_type_idx'] = 5  # mask token
        
        return G_masked
    
    def graph_to_data(self, G):
        """Routes to appropriate conversion based on representation"""
        if self.representation == 'star':
            return self.stargraph_to_data(G)
        elif self.representation == 'component':
            return self.componentgraph_to_data_FIXED(G)  # FIXED VERSION
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
    
    def stargraph_to_data(self, G):
        """Standard star graph representation - UNCHANGED"""
        all_nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        
        node_features = []
        node_types = []
        is_masked = []
        
        for node in all_nodes:
            attr = G.nodes[node]
            feat_dict = attr.get("features", {})
            
            feat = [
                feat_dict.get("node_type_idx", -1),
                feat_dict.get("comp_type_idx", -1),
                feat_dict.get("pin_type_idx", -1)
            ]
            node_features.append(feat)
            
            node_type_map = {"component": 0, "pin": 1, "net": 2, "subcircuit": 3}
            node_types.append(node_type_map.get(attr.get("type"), -1))
            is_masked.append(1 if attr.get('is_masked', False) else 0)
        
        x = torch.tensor(node_features, dtype=torch.long)
        node_type = torch.tensor(node_types, dtype=torch.long)
        mask = torch.tensor(is_masked, dtype=torch.long)
        
        # Build edges
        edge_index = []
        for u, v in G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            is_masked=mask,
            num_nodes=len(all_nodes)
        )
        
        return data
    
    def componentgraph_to_data_FIXED(self, G):
        """
        FIXED VERSION: Instead of extracting only components (which have no edges),
        we keep the FULL graph but modify features to focus on component-level info.
        
        Strategy:
        1. Keep ALL nodes (components, pins, nets)
        2. For non-component nodes, set their features to special "context" tokens
        3. This preserves connectivity while focusing model on component classification
        """
        all_nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        
        node_features = []
        is_masked = []
        is_component = []
        
        for node in all_nodes:
            attr = G.nodes[node]
            feat_dict = attr.get("features", {})
            node_type = attr.get("type")
            
            # For component/subcircuit nodes: use normal features
            if node_type in ["component", "subcircuit"]:
                feat = [
                    feat_dict.get("node_type_idx", -1),
                    feat_dict.get("comp_type_idx", -1),
                    -1  # No pin type for components
                ]
                is_component.append(1)
            else:
                # For pin/net nodes: use special "context" encoding
                # This lets GNN know these exist but focuses on components
                feat = [
                    feat_dict.get("node_type_idx", -1),  # Keep node type
                    5,  # Special token for "non-component"
                    feat_dict.get("pin_type_idx", -1) if node_type == "pin" else -1
                ]
                is_component.append(0)
            
            node_features.append(feat)
            is_masked.append(1 if attr.get('is_masked', False) else 0)
        
        x = torch.tensor(node_features, dtype=torch.long)
        mask = torch.tensor(is_masked, dtype=torch.long)
        comp_mask = torch.tensor(is_component, dtype=torch.long)
        
        # Build edges - KEEP ALL EDGES
        edge_index = []
        for u, v in G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # Fallback for completely disconnected case
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            is_masked=mask,
            is_component=comp_mask,  # New: indicates which nodes are components
            num_nodes=len(all_nodes)
        )
        
        return data
    
    def augment_training_data(self, G, masked_component, comp_type, rng):
        """Light data augmentation - random edge dropout"""
        G_augmented = G.copy()
        
        edges_to_remove = []
        for u, v in G_augmented.edges():
            if u != masked_component and v != masked_component and rng.random() < 0.1:
                edges_to_remove.append((u, v))
        
        G_augmented.remove_edges_from(edges_to_remove)
        return G_augmented


def collate_fegin(batch):
    """Custom collate function that handles graph descriptors correctly"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    descriptors = []
    
    # Extract descriptors and temporarily remove from Data objects
    for data in batch:
        if hasattr(data, "graph_descriptor"):
            desc = data.graph_descriptor
            if desc.ndim != 1:
                raise ValueError(f"Descriptor must be 1D but got shape {desc.shape}")
            descriptors.append(desc)
            delattr(data, "graph_descriptor")

    # Batch all other PyG objects
    from torch_geometric.data import Batch
    batched_data = Batch.from_data_list(batch)

    # Re-insert descriptors as a (batch_size, descriptor_dim) tensor
    if len(descriptors) > 0:
        batched_data.graph_descriptor = torch.stack(descriptors, dim=0)

    return batched_data


if __name__ == "__main__":
    print("Testing FIXED FEGIN Dataset\n")
    
    # Create dummy test data
    test_files = []
    import tempfile
    import shutil
    
    # Create a simple test graph
    G_test = nx.Graph()
    G_test.add_node("R1", type="component", comp_type="R", 
                    features={"node_type_idx": 0, "comp_type_idx": 0, "pin_type_idx": -1})
    G_test.add_node("R1.1", type="pin", component="R1", pin="1",
                    features={"node_type_idx": 1, "comp_type_idx": -1, "pin_type_idx": 0})
    G_test.add_node("R1.2", type="pin", component="R1", pin="2",
                    features={"node_type_idx": 1, "comp_type_idx": -1, "pin_type_idx": 1})
    G_test.add_node("net1", type="net",
                    features={"node_type_idx": 2, "comp_type_idx": -1, "pin_type_idx": -1})
    G_test.add_edges_from([("R1", "R1.1"), ("R1", "R1.2"), ("R1.1", "net1")])
    
    # Save test graph
    temp_dir = tempfile.mkdtemp()
    test_path = os.path.join(temp_dir, "test.gpickle")
    with open(test_path, "wb") as f:
        pickle.dump(G_test, f)
    
    test_files = [(temp_dir, "test.gpickle")]
    
    print("Testing STAR representation:")
    dataset_star = FEGINDatasetFiltered(
        test_files,
        representation='star',
        masks_per_graph=1,
        training=False
    )
    
    sample = dataset_star[0]
    if sample:
        print(f"  Nodes: {sample.num_nodes}, Edges: {sample.edge_index.shape[1]}")
        print(f"  Features: {sample.x.shape}")
        print(f"  Has edges: {sample.edge_index.shape[1] > 0}")
    
    print("\nTesting COMPONENT representation (FIXED):")
    dataset_comp = FEGINDatasetFiltered(
        test_files,
        representation='component',
        masks_per_graph=1,
        training=False
    )
    
    sample = dataset_comp[0]
    if sample:
        print(f"  Nodes: {sample.num_nodes}, Edges: {sample.edge_index.shape[1]}")
        print(f"  Features: {sample.x.shape}")
        print(f"  Has edges: {sample.edge_index.shape[1] > 0} ← FIXED!")
        print(f"  Component mask sum: {sample.is_component.sum().item()}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n✓ FIXED: Component representation now maintains connectivity!")