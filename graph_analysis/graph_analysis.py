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
    avg_shortest_path_length = math.ceil(nx.average_shortest_path_length(G_largest))
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

    return stats


def process_folder(graph_folder):
    graph_stats = []
    for filename in os.listdir(graph_folder):
        if filename.endswith(".gpickle"):
            path = os.path.join(graph_folder, filename)
            with open(path, "rb") as f:
                G = pickle.load(f)
            get_graph_info_detail(G, filename)
            stats = get_graph_info_summary(G, filename)
            graph_stats.append(stats)

    # save graph statistics into a csv file
    df = pd.DataFrame(graph_stats)
    df.to_csv("graph_statistics_align.csv", index=False, sep=";")

if __name__ == "__main__":
    graph_folder = "graph_data_ALIGN"
    process_folder(graph_folder)