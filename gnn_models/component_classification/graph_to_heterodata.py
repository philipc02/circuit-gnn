import torch
from torch_geometric.data import HeteroData
import networkx as nx
import pickle
import os

def graph_to_heterodata(G: nx.Graph) -> HeteroData:
    data = HeteroData()

    # separate nodes by type
    node_types = ["component", "pin", "net", "subcircuit"]
    node_id_map = {t: {} for t in node_types}

    for idx, (n, attr) in enumerate(G.nodes(data=True)):
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
    
        # store label only for component nodes => prediction target!
        if ntype == "component":
            label = torch.tensor([feat_dict.get("comp_type_idx")], dtype=torch.long)

            if "y" not in data["component"]:
                data["component"].y = label
            else:
                data["component"].y = torch.cat([data["component"].y, label], dim=0) # concatenate with existing features of component nodes

    # add edges
    for u, v, eattr in G.edges(data=True):
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

    return data


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