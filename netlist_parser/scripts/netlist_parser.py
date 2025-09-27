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
    "M" : ["drain", "gate", "source"],  # mosfet
    "Q" : ["collector", "base", "emitter"],   # bipolar transistor
    "D": ["anode", "cathode"] # diode
}


def netlist_to_graph_parser(file_path): # port level parser
    parser = SpiceParser(path=file_path)
    circuit = parser.build_circuit() # load circuit object

    G = nx.Graph()
    net_to_terminals = {}   # for creating edges between terminals in same net
    #TODO: graph that depicts nets as nodes

    for element in circuit.element_names:
        comp_type = element[0].upper() # first letter shows which component
        if comp_type not in PIN_ROLES:
            print(f"Element not defined in pin roles: {element}")
            continue
        
        comp = circuit[element]

        pins = PIN_ROLES[comp_type]
        nodes = comp.nodes
        terminal_names = [] #list of terminals for this component

        # drop substrate value (at index 3) and bjt type for now
        if comp_type == "Q" and len(nodes) == 4:
            nodes = nodes[:3]

        # drop bulk value (at index 3) and mosfet type for now
        if comp_type == "M" and len(nodes) == 4:
            nodes = nodes[:3]

        # drop diode model name
        if comp_type == "D" and len(nodes) == 3:
            nodes = nodes[:2]

        for i, net in enumerate(nodes):
            term_name = f"{element}.{pins[i]}"
            terminal_names.append(term_name)
            #TODO: add parameter value as node feature
            G.add_node(term_name,
                        component = str(element),   #converting for safe pickle/unpickle
                        pin = str(pins[i]),
                        net = str(net))
            net_to_terminals.setdefault(str(net), []).append(term_name)  # add mapping between terminal and net its in

        # create edges between all terminals in same net
        for net, terminals in net_to_terminals.items():
            for i in range(len(terminals)):
                for j in range(i + 1, len(terminals)):
                    G.add_edge(terminals[i], terminals[j], net = str(net))

        # add component-internal edges 
        #TODO: check if this correct or misleading for GNN model training
        if len(terminal_names) > 1:
            for i in range(len(terminal_names)):
                for j in range(i + 1, len(terminal_names)):
                    G.add_edge(terminal_names[i], terminal_names[j], component_internal = True)

    return G

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")): #TODO: other netlist formats
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            G = netlist_to_graph_parser(path)

            # save graph in output folder
            graph_filename = os.path.splitext(filename)[0] + ".gpickle" 
            graph_path = os.path.join(output_folder, graph_filename)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)            
            print(f"Saved graph to {graph_path}")

            # draw graph 
            nx.draw(G, with_labels=True, node_size=500)
            plt.show()


if __name__ == "__main__":
    input_folder = "../netlists"
    output_folder = "../graphs"
    process_folder(input_folder, output_folder)
