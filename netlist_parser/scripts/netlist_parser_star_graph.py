import os
import networkx as nx
import pickle
from PySpice.Spice.Parser import SpiceParser
import matplotlib.pyplot as plt
import numpy as np  #TODO: add to requirements.txt

# expand as much as possible
PIN_ROLES = {
    "R" : ["1", "2"], # resistor
    "C" : ["1", "2"],   # capacitor
    "V" : ["pos", "neg"],   # voltage source
    "M" : ["drain", "gate", "source"],  # mosfet
    "Q" : ["collector", "base", "emitter"],   # bipolar transistor
    "D": ["anode", "cathode"] # diode
}

# define different types for node feature vector
NODE_TYPES = ["component", "pin", "net"]

COMPONENT_TYPES = ["R", "C", "V", "M", "Q", "D"]

PIN_TYPES = ["1", "2", "pos", "neg",
             "drain", "gate", "source",
             "collector", "base", "emitter",
             "anode", "cathode"]

def netlist_to_netgraph(file_path, use_star_structure=False):
    parser = SpiceParser(path=file_path)
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

        if use_star_structure:
            # central component node
            G.add_node(element, type="component", comp_type = comp_type, features=encode_node_features("component", comp_type=comp_type))

            # insert explicit pin nodes and connect to component node
            for i, net in enumerate(nets):
                pin_node = f"{element}.{pins[i]}"
                G.add_node(pin_node, type="pin", component=element, pin=pins[i], features=encode_node_features("pin", pin_type=pins[i]))
                # edge from pin to component
                G.add_edge(pin_node, element, kind="component_connection")
                # edge from pin to net node
                G.add_node(str(net), type="net", features=encode_node_features("net"))
                G.add_edge(pin_node, str(net), kind="net_connection")
                    
        else:

            # behaviour from net_as_nodes
            for i, net in enumerate(nets):
                pin_node = f"{element}.{pins[i]}"
                G.add_node(pin_node, type="pin", component=element, pin=pins[i])
                G.add_node(str(net), type="net")
                G.add_edge(pin_node, str(net), kind="net_connection")

            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    G.add_edge(f"{element}.{pins[i]}", f"{element}.{pins[j]}",
                                component=element, pins=(pins[i], pins[j]), kind="component")

    return G

def encode_node_features(node_type, comp_type=None, pin_type=None):
    # one hot encoding for node type
    node_vec = np.zeros(len(NODE_TYPES))
    node_vec[NODE_TYPES.index(node_type)] = 1

    # one hot encoding for component type
    comp_vec = np.zeros(len(COMPONENT_TYPES))
    if node_type == "component" and comp_type in COMPONENT_TYPES:
        comp_vec[COMPONENT_TYPES.index(comp_type)] = 1

    # one hot encoding for pin type
    pin_vec = np.zeros(len(PIN_TYPES))
    if node_type == "pin" and pin_type in PIN_TYPES:
        pin_vec[PIN_TYPES.index(pin_type)] = 1
    # concatenate vectors together -> node feature vector will have length 21
    return np.concatenate([node_vec, comp_vec, pin_vec])

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")):
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            G = netlist_to_netgraph(path, use_star_structure=True)

            # save graph in output folder
            graph_filename = os.path.splitext(filename)[0] + "_star.gpickle"
            graph_path = os.path.join(output_folder, graph_filename)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
            print(f"Saved star graph to {graph_path}")

            pos = nx.kamada_kawai_layout(G)            
            # draw graph
            nx.draw(G, pos=pos, with_labels=True, node_size=500)
            plt.show()


if __name__ == "__main__":
    input_folder = "../netlists"
    output_folder = "../graphs_star"
    process_folder(input_folder, output_folder)