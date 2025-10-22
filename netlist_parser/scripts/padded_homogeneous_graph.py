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

# define different types for node feature vector
NODE_TYPES = ["component", "pin", "net", "subcircuit"]

COMPONENT_TYPES = ["R", "C", "L", "V", "M", "Q", "D", "X", "I", "A", "G"]  # A: arbritrary behaviorial component, G: behaviorial current source

PIN_TYPES = ["1", "2", "pos", "neg",
             "drain", "gate", "source",
             "collector", "base", "emitter",
             "anode", "cathode"]

P_MAX = max(len(pins) for pins in PIN_ROLES.values())

def clean_netlist_file(input_path, cleaned_path):
    with open(input_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if any(param in line.lower() for param in ["rser=", "rpar=", "tol=", "temp=", "ic=", "tc="]):
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

def netlist_to_homograph(file_path, P_MAX=3):
    cleaned_path = file_path + ".clean"
    # clean_netlist_file(file_path, cleaned_path)

    # skip netlists with S elements (unsupported by PySpice) and J/E/B elements (very rare)
    with open(cleaned_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("*")]

    for l in lines:
        if l[0].upper() in {"S", "E", "B", "J"}:
            print(f"Skipping {file_path} due to unsupported/rare element: {l}")
            return None

    parser = SpiceParser(path=cleaned_path)
    try:
        circuit = parser.build_circuit()
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return None

    G = nx.Graph()

    for element in circuit.element_names:
        comp_type = element[0].upper()

        if comp_type not in PIN_ROLES:
            continue

        comp = circuit[element]
        actual_nets = [str(net) for net in comp.nodes]
        actual_pins = PIN_ROLES[comp_type]

        # handle components with extra bulk or substrate nodes
        if comp_type == "Q" and len(actual_nets) == 4:
            actual_nets = actual_nets[:3]
        if comp_type == "M" and len(actual_nets) == 4:
            actual_nets = actual_nets[:3]
        if comp_type == "D" and len(actual_nets) == 3:
            actual_nets = actual_nets[:2]

        G.add_node(element, 
                   type="component",
                   comp_type=comp_type,
                   features=encode_node_features("component", comp_type=comp_type))

        # create exactly P_MAX pins per component and connect to component node
        for i in range(P_MAX):
            # using pin id instead of pin role as node name as we have inactive pin nodes which dont have a role
            pin_id = f"{element}.pin{i}"
            is_active = i < len(actual_pins)
            pin_role = actual_pins[i] if is_active else None
            connected_net = actual_nets[i] if is_active else None
            G.add_node(
                pin_id,
                type="pin",
                component=element,
                pin_index=i,    # index instead of role
                pin_active=int(is_active),  # node has extra information on whether it is active or not
                features=encode_node_features("pin", pin_type=pin_role)
            )
            # edge from pin to component
            G.add_edge(pin_id, element, kind="component_connection")
            # edge from pin to net if active
            if is_active and connected_net:
                G.add_node(connected_net, type="net", features=encode_node_features("net"))
                G.add_edge(pin_id, connected_net, kind="net_connection")

    return G

def encode_node_features(node_type, comp_type=None, pin_type=None):
    # below: encoding node features using one hot vectors and encoding them
    '''
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
    '''
    # second option: store indices, when constructing GNN later in pytorch, we can extract these indices and feed them into embeddings using nn.Embedding
    node_type_idx = NODE_TYPES.index(node_type)

    # component and pin types default to -1 if this is not the node type
    comp_type_idx = COMPONENT_TYPES.index(comp_type) if (comp_type in COMPONENT_TYPES) else -1
    pin_type_idx  = PIN_TYPES.index(pin_type) if (pin_type in PIN_TYPES) else -1

    # store indices
    return {
        "node_type_idx": node_type_idx,
        "comp_type_idx": comp_type_idx,
        "pin_type_idx": pin_type_idx
    }

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    failed = 0
    # mismatched = []
    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")):
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            try:
                G = netlist_to_homograph(path, P_MAX=P_MAX)
                if G is None:
                    failed += 1
                    continue
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")
                failed = failed + 1
                continue

            # save graph in output folder
            graph_filename = os.path.splitext(filename)[0] + "_star_homograph.gpickle"
            graph_path = os.path.join(output_folder, graph_filename)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
            print(f"Saved padded homogeneous star graph to {graph_path}")
            # if not sanity_check(path, G):
            #    mismatched.append(filename)

            pos = nx.kamada_kawai_layout(G)            
            # draw graph
            nx.draw(G, pos=pos, with_labels=True, node_size=500)
            plt.show()
    print(f"Failed to parse {failed} netlists.\n")
    # print(f"Netlists with mismatched component counts: {len(mismatched)}")  # should be 0 now
    # if mismatched:
    #    print("   → " + ", ".join(mismatched))

def sanity_check(file_path, G):
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("*")]
    
    # ignore lines inside subcircuit definitions
    in_subckt = False
    filtered_lines = []
    for l in lines:
        l_upper = l.upper()
        if l_upper.startswith(".SUBCKT"):
            in_subckt = True
            continue
        elif l_upper.startswith(".ENDS"):
            in_subckt = False
            continue
        if not in_subckt:
            filtered_lines.append(l)
    
    # ignoring K (inductance coupling) entries in netlist as we are only looking at physical connections
    ignore_prefixes = ('.', 'K', '+')
    netlist_components = [l for l in filtered_lines if not l[0].upper().startswith(ignore_prefixes)] 

    graph_components = sum(1 for _, d in G.nodes(data=True)
                           if d["type"] in ["component", "subcircuit"])

    print(f"File: {os.path.basename(file_path)}")
    print(f"Netlist component lines: {len(netlist_components)}")
    print(f"Graph component/subcircuit nodes: {graph_components}")

    if len(netlist_components) == graph_components:
        print("Component count matches\n")
        return True
    else:
        print("Mismatch detected\n")
        return False

if __name__ == "__main__":
    input_folder = "../netlists"
    output_folder = "../graphs_star_padded_homogeneous"
    process_folder(input_folder, output_folder)