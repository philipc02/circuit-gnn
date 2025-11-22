# Padded/homogeneous netlist parser, parses only top 5 component types for baseline comparison
# Parses only circuits containing: R, C, V and subcircuits X
# Creates star graphs with P_MAX=2 (since R, C, V all have 2 pins)

import os
import networkx as nx
import pickle
from PySpice.Spice.Parser import SpiceParser
import numpy as np  

# Only top 5 component types
PIN_ROLES = {
    "R" : ["1", "2"], # resistor
    "C" : ["1", "2"],   # capacitor
    "V" : ["pos", "neg"],   # voltage source
}

# define different types for node feature vector
NODE_TYPES = ["component", "pin", "net", "subcircuit"]

COMPONENT_TYPES = ["R", "C", "V", "X"]

PIN_TYPES = ["1", "2", "pos", "neg"]

# P_MAX = 2 since R, C, V all have exactly 2 pins
P_MAX = 2


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

def check_circuit_has_only_allowed_components(file_path):
    cleaned_path = file_path + ".clean"
    
    with open(cleaned_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("*")]
    
    # ignore subcircuit definitions
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
    
    # check for components
    allowed_prefixes = {'R', 'C', 'V', 'X', '.', 'K', '+'}
    
    for line in filtered_lines:
        if not line:
            continue
        first_char = line[0].upper()
        
        if first_char in {'.', 'K', '+'}:
            continue
        
        if first_char not in allowed_prefixes:
            print(f"Contains component type other than R, C, V, X: {first_char}")
            return False
    
    return True


def netlist_to_homograph(file_path, P_MAX=2):
    cleaned_path = file_path + ".clean"
    clean_netlist_file(file_path, cleaned_path)

    # Check if circuit is has only allowed components
    if not check_circuit_has_only_allowed_components(file_path):
        print(f"Skipping {file_path}: contains disallowed component types")
        return None

    # Parse circuit
    parser = SpiceParser(path=cleaned_path)
    try:
        circuit = parser.build_circuit()
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return None

    G = nx.Graph()

    # For tracking
    added_components = set()

    for element in circuit.element_names:
        comp_type = element[0].upper()

        # handle subcircuits like this
        if comp_type == "X":
            comp = circuit[element]
            nets = [str(net) for net in comp.nodes]
            
            G.add_node(
                element,
                type="subcircuit",
                comp_type="X",
                features=encode_node_features("subcircuit", comp_type="X")
            )
            added_components.add('X')

            # For subcircuits, create pins for all connections (no padding, since these won't be masked/predicted by model)
            for i, net in enumerate(nets):
                pin_id = f"{element}.pin{i}"
                G.add_node(
                    pin_id,
                    type="pin",
                    component=element,
                    pin_index=i,
                    pin_active=1,  # All subcircuit pins are active
                    features=encode_node_features("pin")
                )
                G.add_edge(pin_id, element, kind="component_connection")
                G.add_node(str(net), type="net", features=encode_node_features("net"))
                G.add_edge(pin_id, str(net), kind="net_connection")

            continue

        # Skip if not in allowed component types
        if comp_type not in PIN_ROLES:
            print(f"Skipping element {element}: type {comp_type} not allowed")
            continue

        comp = circuit[element]
        actual_nets = [str(net) for net in comp.nodes]
        actual_pins = PIN_ROLES[comp_type]
        
        added_components.add(comp_type)

        # Add component node
        G.add_node(
            element,
            type="component",
            comp_type=comp_type,
            features=encode_node_features("component", comp_type=comp_type)
        )

        # create exactly P_MAX pins per component (with padding)
        for i in range(P_MAX):
            # using pin id instead of pin role as node name as we have inactive pin nodes which dont have a role
            pin_id = f"{element}.pin{i}"
            is_active = i < len(actual_pins)
            pin_role = actual_pins[i] if is_active else None
            connected_net = actual_nets[i] if is_active and i < len(actual_nets) else None
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

    # Verify graph
    if G.number_of_nodes() == 0:
        print(f"Skipping {file_path}: empty graph")
        return None
    
    # Count components
    comp_counts = {ct: 0 for ct in COMPONENT_TYPES}
    for node, attr in G.nodes(data=True):
        if attr.get("type") in ["component"]:
            ct = attr.get("comp_type")
            if ct in comp_counts:
                comp_counts[ct] += 1
    
    print(f"Parsed with components: {comp_counts}")
    
    return G


def encode_node_features(node_type, comp_type=None, pin_type=None):
    # second option: store indices, when constructing GNN later in pytorch, we can extract these indices and feed them into embeddings using nn.Embedding
    node_type_idx = NODE_TYPES.index(node_type)

    # component and pin types default to -1 if this is not the node type
    # ONLY set comp_type_idx for component nodes -> no subcircuit nodes!
    if node_type == "component" and comp_type in COMPONENT_TYPES:
        comp_type_idx = COMPONENT_TYPES.index(comp_type)
    else:
        comp_type_idx = -1 
    # ONLY set pin_type_idx for pin nodes  
    if node_type == "pin" and pin_type in PIN_TYPES:
        pin_type_idx = PIN_TYPES.index(pin_type)
    else:
        pin_type_idx = -1

    # store indices
    return {
        "node_type_idx": node_type_idx,
        "comp_type_idx": comp_type_idx,
        "pin_type_idx": pin_type_idx
    }

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    total_files = 0
    parsed_successfully = 0
    skipped_wrong_components = 0
    failed = 0

    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")):
            total_files += 1
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            
            try:
                G = netlist_to_homograph(path, P_MAX=P_MAX)
                
                if G is None:
                    skipped_wrong_components += 1
                    continue
                
                # save graph in output folder
                graph_filename = os.path.splitext(filename)[0] + "_star_padded_filtered.gpickle"
                graph_path = os.path.join(output_folder, graph_filename)
                with open(graph_path, "wb") as f:
                    pickle.dump(G, f)
                print(f"Saved padded homogeneous star graph to {graph_path}")
                parsed_successfully += 1
                
            except Exception as e:
                print(f"Failed: {e}")
                failed += 1

    print("PARSING SUMMARY")
    print(f"Total files:                {total_files}")
    print(f"Parsed successfully:        {parsed_successfully}")
    print(f"Skipped (wrong components): {skipped_wrong_components}")
    print(f"Failed (parsing errors):    {failed}")
    print(f"Success rate:               {parsed_successfully/total_files*100:.1f}%")


def remove_duplicate_graphs(folder):
    import hashlib

    print(f"Checking for duplicate graphs in {folder}\n")

    unique_hashes = {}
    duplicates = []

    def graph_signature(G):

        def serialize_features(fdict):
            # Convert feature dicts to a sorted tuple of (key, value)
            if isinstance(fdict, dict):
                return tuple(sorted(fdict.items()))
            return fdict

        # Make node and edge lists deterministic
        node_data = sorted(
            (
                d.get("type"),
                d.get("comp_type"),
                d.get("pin_index"),
                d.get("pin_active"),
                serialize_features(d.get("features", {})),
            )
            for _, d in G.nodes(data=True)
        )

        edge_data = sorted(
            (tuple(sorted((u, v))), d.get("kind")) for u, v, d in G.edges(data=True)
        )

        # Hash everything
        m = hashlib.sha256()
        m.update(str(node_data).encode())
        m.update(str(edge_data).encode())
        return m.hexdigest()


    for fname in os.listdir(folder):
        if not fname.endswith(".gpickle"):
            continue
        path = os.path.join(folder, fname)
        with open(path, "rb") as f:
            G = pickle.load(f)

        sig = graph_signature(G)
        if sig in unique_hashes:
            orig = unique_hashes[sig]
            duplicates.append((fname, orig))
        else:
            unique_hashes[sig] = fname

    print(f"Found {len(duplicates)} duplicates out of {len(unique_hashes) + len(duplicates)} total graphs")

    # remove duplicates
    for dup, orig in duplicates:
        os.remove(os.path.join(folder, dup))
        print(f"Removed duplicate: {dup} (matched {orig})")

    print(f"Final dataset: {len(unique_hashes)} unique graphs\n")


def analyze_dataset(folder):
    # Analyze component distribution in filtered dataset
    from collections import Counter
    
    print("DATASET ANALYSIS")
    
    comp_counter = Counter()
    graph_sizes = []
    active_pins = []
    inactive_pins = []
    
    for fname in os.listdir(folder):
        if not fname.endswith(".gpickle"):
            continue
            
        with open(os.path.join(folder, fname), 'rb') as f:
            G = pickle.load(f)
        
        graph_sizes.append(G.number_of_nodes())
        
        for node, attr in G.nodes(data=True):
            if attr.get("type") in ["component", "subcircuit"]:
                comp_type = attr.get("comp_type")
                comp_counter[comp_type] += 1
            elif attr.get("type") == "pin":
                if attr.get("pin_active") == 1:
                    active_pins.append(1)
                else:
                    inactive_pins.append(1)
    
    print("Component Type Distribution:")
    total = sum(comp_counter.values())
    for comp_type in COMPONENT_TYPES:
        count = comp_counter[comp_type]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {comp_type:5s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"Total components: {total}")
    print(f"Total graphs: {len(graph_sizes)}")
    print(f"Avg nodes per graph: {np.mean(graph_sizes):.1f}")
    print(f"Avg components per graph: {total / len(graph_sizes):.1f}")
    print(f"Pin Statistics:")
    print(f"  Active pins:   {len(active_pins)}")
    print(f"  Inactive pins: {len(inactive_pins)}")
    print(f"  Padding ratio: {len(inactive_pins)/(len(active_pins)+len(inactive_pins))*100:.1f}%")


if __name__ == "__main__":
    print("Netlist parser for padded netlists running...")
    
    input_folder = "../netlists"
    output_folder = "../graphs_star_padded_filtered"
    process_folder(input_folder, output_folder)
    remove_duplicate_graphs(output_folder)
    analyze_dataset(output_folder)