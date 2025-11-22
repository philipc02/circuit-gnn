# Dataset loader for FEGIN component classification with filtered top-4 component types
# To work with: R, C, V, X

import torch
from torch_geometric.data import Data, Dataset, Batch
import networkx as nx
import pickle
import os
from graph_descriptors import GraphDescriptorCache

# Only the 4 component types we want to classify (including X/subcircuits as baseline is also predicting 'sub elements')
COMPONENT_TYPES = ["R", "C", "V", "X"]  # 4-class problem


class FEGINDatasetFiltered(Dataset):
    # Dataset for FEGIN component classification with filtered types
    # Prredicts R, C, V, X

    
    def __init__(self, fold_dir, split, representation='star', 
                 mask_strategy='keep_pins', n_eigenvalues=10, dgsd_bins=10):
        """
        Args:
            fold_dir: Path to fold directory
            split: 'train', 'val', 'test'
            representation: 'star', 'component'
            mask_strategy: 'keep_pins', 'remove_pins'
            n_eigenvalues: Number of eigenvalues for NetLSD
            dgsd_bins: Number of bins for DGSD
        """
        super().__init__()
        
        self.folder = os.path.join(fold_dir, split)
        self.representation = representation
        self.mask_strategy = mask_strategy
        self.n_eigenvalues = n_eigenvalues
        self.dgsd_bins = dgsd_bins
        
        # Load all graph files
        self.files = sorted([f for f in os.listdir(self.folder) 
                           if f.endswith(".gpickle")])
        
        # Initialize descriptor cache
        self.descriptor_cache = GraphDescriptorCache(n_eigenvalues, dgsd_bins)
        
        print(f"Loaded {len(self.files)} graphs from {self.folder}")
    
    def len(self):
        return len(self.files)
    
    def get(self, idx):
        """
        Load graph, create masked version, convert to PyG Data
        Only mask R, C, V components and X subcircuits
        """
        # Load graph
        graph_path = os.path.join(self.folder, self.files[idx])
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        # Get R, C, V components AND X subcircuits
        component_nodes = [n for n, attr in G.nodes(data=True) 
                          if (attr.get("type") == "component" and attr.get("comp_type") in ["R", "C", "V"])
                          or (attr.get("type") == "subcircuit" and attr.get("comp_type") == "X")]
        
        if len(component_nodes) == 0:
            return None
        
        # Randonly select a component to mask
        import random
        masked_component = random.choice(component_nodes)
        comp_type = G.nodes[masked_component].get("comp_type")
        
        # Check validity
        if comp_type not in COMPONENT_TYPES:
            return None
        
        # Create masked graph
        G_masked = self.create_masked_graph(G, masked_component)
        
        # Convert to PyG Data
        data = self.graph_to_data(G_masked)
        
        # add graph descriptors
        descriptor = self.descriptor_cache.get_or_compute(self.files[idx], G_masked)
        data.graph_descriptor = descriptor
        
        # add label
        data.y = torch.tensor(COMPONENT_TYPES.index(comp_type), dtype=torch.long)
        data.masked_component = masked_component
        data.graph_id = self.files[idx]
        
        return data
    
    def create_masked_graph(self, G, masked_component):
        G_masked = G.copy()
        
        if self.mask_strategy == 'remove_pins':
            # Remove component and pins entirely
            nodes_to_remove = [masked_component]
            
            for neighbor in G.neighbors(masked_component):
                node_attr = G.nodes[neighbor]
                if (node_attr.get("type") == "pin" and 
                    node_attr.get("component") == masked_component):
                    nodes_to_remove.append(neighbor)
            
            G_masked.remove_nodes_from(nodes_to_remove)
            
        elif self.mask_strategy == 'keep_pins':
            # Keep pins but mask features
            G_masked.nodes[masked_component]['is_masked'] = True
            
            if 'features' in G_masked.nodes[masked_component]:
                G_masked.nodes[masked_component]['features']['comp_type_idx'] = -1
            
            # Mask pins
            for neighbor in G.neighbors(masked_component):
                node_attr = G.nodes[neighbor]
                if (node_attr.get("type") == "pin" and 
                    node_attr.get("component") == masked_component):
                    G_masked.nodes[neighbor]['is_masked'] = True
                    if 'features' in G_masked.nodes[neighbor]:
                        G_masked.nodes[neighbor]['features']['pin_type_idx'] = -1
        
        return G_masked
    
    def graph_to_data(self, G):
        if self.representation == 'star':
            return self.stargraph_to_data(G)
        elif self.representation == 'component':
            return self.componentgraph_to_data(G)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
    
    def stargraph_to_data(self, G):
        all_nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        
        # Build node features
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
            
            # Node type
            node_type_map = {"component": 0, "pin": 1, "net": 2, "subcircuit": 3}
            node_types.append(node_type_map.get(attr.get("type"), -1))
            
            # Masked indicator
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
    
    def componentgraph_to_data(self, G):
        # Include both components and subcircuits
        component_nodes = [n for n, attr in G.nodes(data=True) 
                          if attr.get("type") in ["component", "subcircuit"]]
        
        if len(component_nodes) == 0:
            return Data(x=torch.empty((0, 3), dtype=torch.long),
                       edge_index=torch.empty((2, 0), dtype=torch.long),
                       num_nodes=0)
        
        node_to_idx = {n: i for i, n in enumerate(component_nodes)}
        
        node_features = []
        is_masked = []
        
        for node in component_nodes:
            attr = G.nodes[node]
            feat_dict = attr.get("features", {})
            
            feat = [
                feat_dict.get("node_type_idx", -1),
                feat_dict.get("comp_type_idx", -1),
                -1  # No pin type
            ]
            node_features.append(feat)
            is_masked.append(1 if attr.get('is_masked', False) else 0)
        
        x = torch.tensor(node_features, dtype=torch.long)
        mask = torch.tensor(is_masked, dtype=torch.long)
        
        # Build edges
        edge_index = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
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
            is_masked=mask,
            num_nodes=len(component_nodes)
        )
        
        return data

def collate_fegin(batch):
    """Custom collate function that handles graph descriptors correctly."""
    print("collate_fegin called for batch of length:", len(batch))
    # Remove None samples
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    descriptors = []
    
    # --- 1) Extract descriptors and temporarily remove from Data objects ---
    for data in batch:
        if hasattr(data, "graph_descriptor"):
            # Ensure this is a 1D tensor
            desc = data.graph_descriptor
            if desc.ndim != 1:
                raise ValueError(f"Descriptor must be 1D but got shape {desc.shape}")
            descriptors.append(desc)
            delattr(data, "graph_descriptor")

    # --- 2) Batch all other PyG objects ---
    from torch_geometric.data import Batch
    batched_data = Batch.from_data_list(batch)

    # --- 3) Re-insert descriptors as a (batch_size, descriptor_dim) tensor ---
    if len(descriptors) > 0:
        batched_data.graph_descriptor = torch.stack(descriptors, dim=0)

    print("BATCH DESCRIPTOR SHAPE:", batched_data.graph_descriptor.shape)


    return batched_data



if __name__ == "__main__":
    print("Testing Filtered FEGIN Dataset\n")
    
    # Test dataset
    print("Loading Dataset")
    
    dataset = FEGINDatasetFiltered(
        fold_dir="../../data/data_kfold_filtered/fold_0",
        split="train",
        representation='star',
        mask_strategy='keep_pins'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    if sample is not None:
        print(f"Sample data:")
        print(f"  Nodes: {sample.num_nodes}")
        print(f"  Edges: {sample.edge_index.size(1)}")
        print(f"  Features: {sample.x.shape}")
        print(f"  Descriptor: {sample.graph_descriptor.shape}")
        print(f"  Label: {sample.y.item()} ({COMPONENT_TYPES[sample.y.item()]})")

    print("Dataset ready!")
