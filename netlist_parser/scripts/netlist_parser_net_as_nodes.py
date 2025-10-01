import os
import networkx as nx
import pickle
from PySpice.Spice.Parser import SpiceParser
import matplotlib.pyplot as plt
import numpy as np

# expand as much as possible
PIN_ROLES = {
    "R" : ["1", "2"], # resistor
    "C" : ["1", "2"],   # capacitor
    "L" : ["1", "2"],       # inductor
    "V" : ["pos", "neg"],   # voltage source
    "I": ["pos", "neg"],    # current source
    "M" : ["drain", "gate", "source"],  # mosfet
    "Q" : ["collector", "base", "emitter"],   # bipolar transistor
    "D": ["anode", "cathode"] # diode
}

NODE_TYPES = ["pin", "net"]
PIN_TYPES = ["1", "2", "pos", "neg",
             "drain", "gate", "source",
             "collector", "base", "emitter",
             "anode", "cathode"]
# wiring edge is between terminals of different components, internal is between terminals of the same component
EDGE_TYPES = ["wiring", "internal"]

def clean_netlist_file(input_path, cleaned_path):
    with open(input_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if any(param in line.lower() for param in ["rser=", "rpar="]):
            tokens = line.split()
            # keep element name, node connections, first numeric/model token
            keep = []
            for tok in tokens:
                if "=" in tok:  # stop before params
                    break
                keep.append(tok)
            cleaned_lines.append(" ".join(keep) + "\n")
        else:
            cleaned_lines.append(line)

    with open(cleaned_path, "w") as f:
        f.writelines(cleaned_lines)

def netlist_to_netgraph(file_path, use_pin_nodes=False):
    # clean netlist first
    cleaned_path = file_path + ".clean"
    clean_netlist_file(file_path, cleaned_path)
    parser = SpiceParser(path=cleaned_path)
    circuit = parser.build_circuit()

    G = nx.Graph()

    for element in circuit.element_names:
        comp_type = element[0].upper()
        if comp_type not in PIN_ROLES:
            print(f"Element not defined in pin roles: {element}")
            continue

        comp = circuit[element]
        pins = PIN_ROLES[comp_type]
        nets = [str(net) for net in comp.nodes]

        # drop substrate value (at index 3) and bjt type for now
        if comp_type == "Q" and len(nets) == 4:
            nets = nets[:3]

        # drop bulk value (at index 3) and mosfet type for now
        if comp_type == "M" and len(nets) == 4:
            nets = nets[:3]

        # drop diode model name
        if comp_type == "D" and len(nets) == 3:
            nets = nets[:2]

        if use_pin_nodes:
            # insert explicit pin nodes
            for i, net in enumerate(nets):
                pin_node = f"{element}.{pins[i]}"
                G.add_node(pin_node, type="pin", component=element, pin=pins[i], features = encode_node_features("pin", pin_type=pins[i]))
                G.add_node(str(net), type="net", features = encode_node_features("net"))
                G.add_edge(pin_node, str(net), kind="wiring", features = encode_edge_features("wiring"))

            # add virtual edge connecting all pins of the same component
            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    G.add_edge(f"{element}.{pins[i]}", f"{element}.{pins[j]}",
                               component=element, pins=(pins[i], pins[j]), kind="internal", features = encode_edge_features("internal"))
                    
        else:

            # add nets as graph nodes
            # no features
            for net in nets:
                if not G.has_node(net):
                    G.add_node(net, is_net=True)

            # for each pair of nets add edge representing component
            if len(nets) == 2:
                G.add_edge(nets[0], nets[1],
                        component=element,
                        pins=(pins[0], pins[1]))
            else:
                # for components with multiple terminals (mosfet, bipolar transistor)
                for i in range(len(nets)):
                    for j in range(i+1, len(nets)):
                        G.add_edge(nets[i], nets[j],
                                component=element,
                                pins=(pins[i], pins[j]))

    return G

def encode_node_features(node_type, pin_type=None):
    # one hot encoding for node type
    node_vec = np.zeros(len(NODE_TYPES))
    node_vec[NODE_TYPES.index(node_type)] = 1

    # one hot encoding for pin type
    pin_vec = np.zeros(len(PIN_TYPES))
    if node_type == "pin" and pin_type in PIN_TYPES:
        pin_vec[PIN_TYPES.index(pin_type)] = 1

    # concatenate vectors together -> node feature vector will have length 14
    return np.concatenate([node_vec, pin_vec])  # length = 2 + len(PIN_TYPES)


def encode_edge_features(edge_type):
    # one hot encoding for edge type
    edge_vec = np.zeros(len(EDGE_TYPES))
    edge_vec[EDGE_TYPES.index(edge_type)] = 1
    # edge feature vector length 2
    return edge_vec


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")):
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            G = netlist_to_netgraph(path, use_pin_nodes=True)

            # save graph in output folder
            graph_filename = os.path.splitext(filename)[0] + "_nets.gpickle"
            graph_path = os.path.join(output_folder, graph_filename)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
            print(f"Saved net-based graph to {graph_path}")

            pos = nx.spring_layout(G, seed=42)
            # draw graph
            nx.draw(G, pos=pos, with_labels=True, node_size=500)
            plt.show()



if __name__ == "__main__":
    input_folder = "../netlists"
    output_folder = "../graphs_nets"
    process_folder(input_folder, output_folder)