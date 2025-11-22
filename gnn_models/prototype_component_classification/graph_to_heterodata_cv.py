import torch
from torch_geometric.data import HeteroData
import networkx as nx
import pickle
import os
import random

COMPONENT_TYPES = ["R", "C", "L", "V", "M", "Q", "D", "I"]

def mask_graph(G: nx.Graph, num_masks_per_graph=4):
    masked_graphs = []
    component_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "component"]
    components_to_mask = min(len(component_nodes), num_masks_per_graph)

    if components_to_mask != num_masks_per_graph:
        masked_components = component_nodes
    else:
        masked_components = random.sample(component_nodes, num_masks_per_graph)

    for component in masked_components:
        G_masked = G.copy()
        pin_nodes = []
        for neighbor in G.neighbors(component):
            if G.nodes[neighbor].get("type") == "pin" and G.nodes[neighbor].get("component") == component:
                pin_nodes.append(neighbor)

        nodes_to_remove = [component] + pin_nodes
        G_masked.remove_nodes_from(nodes_to_remove)
        G_masked.graph['masked_component'] = component
        G_masked.graph['masked_component_type'] = G.nodes[component].get("comp_type")
        G_masked.graph['masked_pins'] = pin_nodes

        masked_graphs.append(G_masked)
    return masked_graphs

def graph_to_heterodata(G: nx.Graph, num_masks=4) -> HeteroData:
    masked_graphs = mask_graph(G, num_masks_per_graph=num_masks)
    hetero_data_list = []

    for graph in masked_graphs:
        data = HeteroData()

        node_types = ["component", "pin", "net", "subcircuit"]
        node_id_map = {t: {} for t in node_types}

        for idx, (n, attr) in enumerate(graph.nodes(data=True)):
            ntype = attr["type"]
            node_id_map[ntype][n] = len(node_id_map[ntype])

            feat_dict = attr.get("features", {})
            feat_tensor = torch.tensor([
                feat_dict.get("node_type_idx", -1),
                feat_dict.get("comp_type_idx", -1),
                feat_dict.get("pin_type_idx", -1)
            ], dtype=torch.long).unsqueeze(0)

            if "x" not in data[ntype]:
                data[ntype].x = feat_tensor
            else:
                data[ntype].x = torch.cat([data[ntype].x, feat_tensor], dim=0)

        for u, v, eattr in graph.edges(data=True):
            etype = eattr["kind"]
            src_type = graph.nodes[u]["type"]
            dst_type = graph.nodes[v]["type"]
            edge_type = (src_type, etype, dst_type)
            rev_edge_type = (dst_type, etype, src_type)

            src = node_id_map[src_type][u]
            dst = node_id_map[dst_type][v]
            edge_index = torch.tensor([[src], [dst]], dtype=torch.long)
            rev_edge_index = torch.tensor([[dst], [src]], dtype=torch.long)

            if "edge_index" not in data[edge_type]:
                data[edge_type].edge_index = edge_index
            else:
                data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, edge_index], dim=1)

            if "edge_index" not in data[rev_edge_type]:
                data[rev_edge_type].edge_index = rev_edge_index
            else:
                data[rev_edge_type].edge_index = torch.cat([data[rev_edge_type].edge_index, rev_edge_index], dim=1)

        comp_label = graph.graph['masked_component_type']
        data.target_comp_type = torch.tensor(COMPONENT_TYPES.index(comp_label), dtype=torch.long)
        data.masked_comp = graph.graph['masked_component']
        data.original_graph_id = G.graph.get('original_id', 'unknown')  # Store original graph ID
        hetero_data_list.append(data)

    return hetero_data_list

def create_cross_validation_data():
    base_input_folder = "../../data/data_kfold"
    base_output_folder = "../../data/data_cross_validation_heterodata"
    
    # Process each fold
    for fold_idx in range(5):
        print(f"Processing fold {fold_idx}...")
        
        fold_input_folder = os.path.join(base_input_folder, f"fold_{fold_idx}")
        fold_output_folder = os.path.join(base_output_folder, f"fold_{fold_idx}")
        
        splits = ["train", "val", "test"]
        
        for split in splits:
            in_dir = os.path.join(fold_input_folder, split)
            out_dir = os.path.join(fold_output_folder, split)
            os.makedirs(out_dir, exist_ok=True)

            if not os.path.exists(in_dir):
                print(f"Warning: {in_dir} does not exist, skipping...")
                continue

            for fname in os.listdir(in_dir):
                if fname.endswith(".gpickle"):
                    with open(os.path.join(in_dir, fname), "rb") as f:
                        G = pickle.load(f)
                    
                    # Add original graph ID for tracking
                    G.graph['original_id'] = fname
                    
                    data_list = graph_to_heterodata(G)
                    for i, data in enumerate(data_list):
                        torch.save(data, os.path.join(out_dir, fname.replace(".gpickle", f"_masked_{i}.pt")))

        print(f"Fold {fold_idx} completed.")

if __name__ == "__main__":
    create_cross_validation_data()
    print("Cross-validation datasets created successfully!")