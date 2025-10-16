import networkx as nx
import os
import pickle
from collections import Counter
import math
import pandas as pd

def get_graph_info_detail(graph, filename):

    print("Graph analysis for ", filename)

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Node degree:", dict(graph.degree()))
    
    # connectivity ? 

    # 1. Definition: the average shortest path of all the connected node pairs
    # 2. Interpretation: On average, how far away the nodes are from each other.
    largest_cc = max(nx.connected_components(graph), key=len)
    G_largest = graph.subgraph(largest_cc)
    avg_shortest_path_length = nx.average_shortest_path_length(G_largest)
    print(f"Average shortest path length (largest component): {avg_shortest_path_length}")


    # 1. Definition: the ratio between the count of existing edges and all the possible edges’ count
    # the higher the ratio, the higher the complexity of the graph
    # the denser the graph, the more likely thatr more edge connections will be lost with the loss of a node (for training: remove node)
    density = nx.density(graph)
    print(f"Density: {density: .4f}") # round to four digits after comma

    # Counter object to count the node types (component, pin, net, subcircuit)
    node_types = Counter(nx.get_node_attributes(graph, "type").values())
    print(f"Node type counts: {dict(Counter(node_types))}")
    # same for component types (resistor, capacitor etc)
    comp_types = [graph.nodes[n].get("comp_type") for n in graph.nodes if graph.nodes[n].get("type") == "component"] # create indexed list of component types
    comp_types = [c for c in comp_types if c is not None] # remove empty entries (node is not a component)
    print(f"Component type counts: {dict(Counter(comp_types))}\n\n")


    # hubs -> which component type (question: what if a net node is a hub?)
    # hub here is node with the most edge connections -> component has max 3 pins, hub more likely to be net or subcircuit node => doesn't make sense to find component type
    # hubs  = sorted(dict(graph.degree()).items(), key=lambda x: x[1], reverse=True)[:3] # pick top 3 hub nodes
    # print(f"Top three hubs: ")
    # for node, deg in hubs:
        # ntype = graph.nodes[node].get("type")
        # ctype = graph.nodes[node].get("comp_type")
        # print(f"{node} degree={deg} type={ntype} comp_type={ctype}")

def get_graph_info_summary(graph, filename):

    largest_cc = max(nx.connected_components(graph), key=len)
    G_largest = graph.subgraph(largest_cc)
    avg_shortest_path_length = math.ceil(nx.average_shortest_path_length(G_largest))

    stats = {
        "filename": filename,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "node_degrees": dict(graph.degree()),
        "average_shortest_path_length": avg_shortest_path_length,
        "density": nx.density(graph),
    }

    node_types = Counter(nx.get_node_attributes(graph, "type").values())
    for k, v in node_types.items():
        stats[f"num_nodes_{k}"] = v

    comp_types = [graph.nodes[n].get("comp_type") for n in graph.nodes if graph.nodes[n].get("type") == "component"]
    comp_types = [c for c in comp_types if c is not None]
    comp_type_counts = Counter(comp_types)
    for k, v in comp_type_counts.items():
        stats[f"num_components_{k}"] = v

    # find ratios
    num_components = node_types.get("component", 0)
    num_pins = node_types.get("pin", 0)
    num_nets = node_types.get("net", 0)

    # avoid division by zero
    if num_components > 0:
        stats["net_to_component_ratio"] = num_nets / num_components
        stats["pin_to_component_ratio"] = num_pins / num_components
    else:
        stats["net_to_component_ratio"] = 0
        stats["pin_to_component_ratio"] = 0

    # average number of connections per net
    net_nodes = [n for n in graph.nodes if graph.nodes[n].get("type") == "net"]
    if len(net_nodes) > 0:
        avg_conn_per_net = sum(dict(graph.degree(net_nodes)).values()) / len(net_nodes)
    else:
        avg_conn_per_net = 0
    stats["avg_connections_per_net"] = avg_conn_per_net

    return stats


def process_folder(graph_folder):
    graph_stats = []
    # collect global counts for each component type across all graphs, useful for later weighting to counter dataset imbalances
    global_comp_counter = Counter()
    for filename in os.listdir(graph_folder):
        if filename.endswith(".gpickle"):
            path = os.path.join(graph_folder, filename)
            with open(path, "rb") as f:
                G = pickle.load(f)
            get_graph_info_detail(G, filename)  # print details onto console
            stats = get_graph_info_summary(G, filename)
            graph_stats.append(stats)

            comp_types = [
                G.nodes[n].get("comp_type")
                for n in G.nodes
                if G.nodes[n].get("type") == "component" and G.nodes[n].get("comp_type") is not None
            ]
            global_comp_counter.update(comp_types)

    # save graph statistics into a csv file
    df = pd.DataFrame(graph_stats)
    df = pd.DataFrame(graph_stats).fillna(0)  # fill missing columns with 0
    # dataset-level summaries
    numeric_cols = ["num_nodes", "num_edges", "average_shortest_path_length", "density", "net_to_component_ratio", "pin_to_component_ratio", "avg_connections_per_net"]
    summary = df[numeric_cols].mean().to_frame().T

    node_type_cols = [c for c in df.columns if c.startswith("num_nodes_")]
    comp_type_cols = [c for c in df.columns if c.startswith("num_components_")]

    node_type_avg = pd.DataFrame([df[node_type_cols].mean()]) if node_type_cols else pd.DataFrame()
    comp_type_avg = pd.DataFrame([df[comp_type_cols].mean()]) if comp_type_cols else pd.DataFrame()
    summary_full = pd.concat([summary, node_type_avg, comp_type_avg], axis=1)
    summary_full.index = ["mean"]

    total_components = sum(global_comp_counter.values())
    freq_rows = []
    for comp_type, count in global_comp_counter.items():
        freq_rows.append({
            "component_type": comp_type,
            "count": count,
            "relative_frequency_percent": (count / total_components * 100) if total_components > 0 else 0
        })
    freq_df = pd.DataFrame(freq_rows).sort_values("count", ascending=False)

    with open("graph_statistics_summary.csv", "w", encoding="utf-8") as f:
        summary_full.to_csv(f, sep=";", decimal=",")
        f.write("\n\n# Global component class frequencies\n")
        freq_df.to_csv(f, sep=";", index=False, decimal=",")
    # df.to_csv("graph_statistics.csv", index=False, sep=";")

if __name__ == "__main__":
    #graph_folder = r"C:\Users\chris\OneDrive\Desktop\Chrissa\University\Sem 7\Bachelor's Thesis\Literature & resource review\Circuit-Completion-Using-GNNs\component classification\data\processed_graphs"
    graph_folder = "graph_data"
    process_folder(graph_folder)