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
    components_to_mask = min(len(component_nodes), num_masks_per_graph)   # in case graph has less componentd

    if components_to_mask != num_masks_per_graph:
        masked_components = component_nodes  # all nodes will be masked once
    else:
        masked_components = random.sample(component_nodes, num_masks_per_graph)  # randomly choose 4 components to hide (connected pin will be handled afterwards)

    for component in masked_components:
        G_masked = G.copy()  # work on copy of original graph
        pin_nodes = []
        # corresponding pin nodes
        for neighbor in G.neighbors(component):
            if G.nodes[neighbor].get("type") == "pin" and G.nodes[neighbor].get("component") == component:
                pin_nodes.append(neighbor)

        nodes_to_remove = [component] + pin_nodes  # list of all node to be removed
        G_masked.remove_nodes_from(nodes_to_remove)  # edges connected to these nodes are removed too
        # add as attributes component, matching pin nodes and comp type of masked node for labeling
        G_masked.graph['masked_component'] = component
        G_masked.graph['masked_component_type'] = G.nodes[component].get("comp_type")
        G_masked.graph['masked_pins'] = pin_nodes

        # collect all masked graph versions in this list
        masked_graphs.append(G_masked)
    return masked_graphs


def graph_to_heterodata(G: nx.Graph, num_masks=4) -> HeteroData:
    masked_graphs = mask_graph(G, num_masks_per_graph=num_masks)
    hetero_data_list = []

    for graph in masked_graphs:
        data = HeteroData()

        # separate nodes by type
        node_types = ["component", "pin", "net", "subcircuit"]
        node_id_map = {t: {} for t in node_types}

        for idx, (n, attr) in enumerate(graph.nodes(data=True)):
            ntype = attr["type"]
            # define unique index for node using length of list of nodes of this type and add to map indexed with node
            node_id_map[ntype][n] = len(node_id_map[ntype])

            # extract feature indices
            feat_dict = attr.get("features", {})
            feat_tensor = torch.tensor([
                feat_dict.get("node_type_idx", -1),
                feat_dict.get("comp_type_idx", -1),
                feat_dict.get("pin_type_idx", -1)
            ], dtype=torch.long).unsqueeze(0) # unsqueeze(0) inserts a new dimension at index 0: (3,) -> (1, 3) ("one node with three features")

            if "x" not in data[ntype]:
                data[ntype].x = feat_tensor
            else:
                data[ntype].x = torch.cat([data[ntype].x, feat_tensor], dim=0)  # concatenate with existing features of nodes of this type

        # add edges
        for u, v, eattr in graph.edges(data=True):
            # to create tuple containing information on edge
            etype = eattr["kind"]
            src_type = G.nodes[u]["type"]
            dst_type = G.nodes[v]["type"]
            edge_type = (src_type, etype, dst_type)
            rev_edge_type = (dst_type, etype, src_type)

            # retrieve unique node index from map and create tensor
            src = node_id_map[src_type][u]
            dst = node_id_map[dst_type][v]
            edge_index = torch.tensor([[src], [dst]], dtype=torch.long)
            rev_edge_index = torch.tensor([[dst], [src]], dtype=torch.long)

            # dimension will be (2, number of source/destination nodes) with 2 indicating the two lists storing source and destination nodes
            if "edge_index" not in data[edge_type]:
                data[edge_type].edge_index = edge_index
            else:
                data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, edge_index], dim=1) # concatenate with existing features of edges of this type (same source and destination type as well as same edge type (component to pin or pin to net))

            # add reversed relation as well (our graphs are undirected)
            if "edge_index" not in data[rev_edge_type]:
                data[rev_edge_type].edge_index = rev_edge_index
            else:
                data[rev_edge_type].edge_index = torch.cat([data[rev_edge_type].edge_index, rev_edge_index], dim=1)

        # this is what model has to guess based on 
        comp_label = graph.graph['masked_component_type']
        data.target_comp_type = torch.tensor(COMPONENT_TYPES.index(comp_label), dtype=torch.long)
        data.masked_comp = graph.graph['masked_component']
        hetero_data_list.append(data)

    return hetero_data_list


def load_hetero_graphs(folder_path):
    hetero_graphs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".gpickle"):
            with open(os.path.join(folder_path, fname), "rb") as f:
                G = pickle.load(f)
            hetero_graphs.append(graph_to_heterodata(G))
    return hetero_graphs

if __name__ == "__main__":
    graph_folder = "../../data/data_test"
    hetero_graphs = load_hetero_graphs(graph_folder)
    data = hetero_graphs[0]
    print(data)               