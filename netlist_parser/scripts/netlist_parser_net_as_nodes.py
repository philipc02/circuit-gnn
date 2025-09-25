import os
import networkx as nx
import pickle
from PySpice.Spice.Parser import SpiceParser
import matplotlib.pyplot as plt

# expand as much as possible
PIN_ROLES = {
    "R" : ["1", "2"], # resistor
    "C" : ["1", "2"],   # capacitor
    "V" : ["pos", "neg"],   # voltage source
    "M" : ["d", "g", "s"],  # mosfet
    "Q" : ["c", "b", "e"],   # bipolar transistor
    "D": ["anode", "cathode"] # diode
}

def netlist_to_netgraph(file_path, use_pin_nodes=False):
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

        if use_pin_nodes:
            # insert explicit pin nodes
            for i, net in enumerate(nets):
                pin_node = f"{element}.{pins[i]}"
                G.add_node(pin_node, type="pin", component=element, pin=pins[i])
                G.add_node(str(net), type="net")
                G.add_edge(pin_node, str(net), kind="net_connection")

            # add virtual edge connecting all pins of the same component
            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    G.add_edge(f"{element}.{pins[i]}", f"{element}.{pins[j]}",
                               component=element, pins=(pins[i], pins[j]), kind="component")
                    
        else:

            # add nets as graph nodes
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
            nx.draw(G, with_labels=True, node_size=500)
            plt.show()


if __name__ == "__main__":
    input_folder = "../netlists"
    output_folder = "../graphs_nets"
    process_folder(input_folder, output_folder)