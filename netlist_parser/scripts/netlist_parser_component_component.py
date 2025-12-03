# Parser for circuit representation with only component nodes (no pin or net nodes): component - component
# Edges: direct connections between components sharing nets
import os
import networkx as nx
import pickle
from PySpice.Spice.Parser import SpiceParser
import numpy as np 

COMPONENT_TYPES = ["R", "C", "V", "X"]

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
    # Read netlist
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
    allowed_prefixes = {'R', 'C', 'V', 'X', '.', 'K', '+'}  # . for directives, K for coupling, + for continuation
    
    for line in filtered_lines:
        if not line:
            continue
        first_char = line[0].upper()
        
        # Skip directives and special lines
        if first_char in {'.', 'K', '+'}:
            continue
        
        # Check if component type is allowed
        if first_char not in allowed_prefixes:
            print(f"Contains component type other than R, C, V, X: {first_char}")
            return False
    
    return True


def netlist_to_component_component_graph(file_path):
    # clean netlist first
    cleaned_path = file_path + ".clean"
    clean_netlist_file(file_path, cleaned_path)

    # Check if circuit contains only allowed components
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
    
    # Dictionary to track which components are connected to each net
    net_to_components = {}
    
    # First pass (collect all components and their nets)
    for element in circuit.element_names:
        comp_type = element[0].upper()

        if comp_type not in COMPONENT_TYPES:
            print(f"Skipping {element}: not in allowed component types")
            continue

        comp = circuit[element]
        nets = [str(net) for net in comp.nodes]
        
        # Add component node
        G.add_node(element, type="component", comp_type = comp_type, features=encode_node_features(comp_type))
        
        # Track net connections
        for net in nets:
            if net not in net_to_components:
                net_to_components[net] = []
            net_to_components[net].append(element) # All elements in this list need to be connected later
    
    # Second pass (connect components that share nets)
    for net, components in net_to_components.items():
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                comp1 = components[i]
                comp2 = components[j]
                G.add_edge(comp1, comp2, kind="direct_connection", net=net, weight=1.0)
    
    # Remove isolated nodes (shouldnt happen, but safety check)
    G.remove_nodes_from(list(nx.isolates(G)))

    # verify graph has components
    if G.number_of_nodes() == 0:
        print(f"Skipping {file_path}: empty graph after filtering")
        return None
    
    print(f"Component-component graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def encode_node_features(comp_type):
    # store indices, when constructing GNN later in pytorch, we can extract these indices and feed them into embeddings using nn.Embedding
    return {
        "comp_type_idx": COMPONENT_TYPES.index(comp_type)
    }

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    total_files = 0
    parsed_successfully = 0
    skipped_wrong_components = 0
    failed= 0

    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")):
            total_files += 1
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            try:
                G = netlist_to_component_component_graph(path)
                if G is None:
                    skipped_wrong_components += 1
                    continue
                    
                # save graph in output folder
                graph_filename = os.path.splitext(filename)[0] + "_component_component.gpickle"
                graph_path = os.path.join(output_folder, graph_filename)
                with open(graph_path, "wb") as f:
                    pickle.dump(G, f)
                print(f"Saved to {graph_path}")
                parsed_successfully += 1
                
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")
                failed += 1

    print("PARSING SUMMARY\n")
    print(f"Total files:               {total_files}")
    print(f"Parsed successfully:       {parsed_successfully}")
    print(f"Skipped (wrong components): {skipped_wrong_components}")
    print(f"Failed (parsing errors):   {failed}")
    print(f"Success rate:              {parsed_successfully/total_files*100:.1f}%")


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

    print(f"\nFinal dataset: {len(unique_hashes)} unique graphs")


def analyze_dataset(folder):
    # Analyze component distribution in filtered dataset
    from collections import Counter
    
    print("DATASET ANALYSIS\n")
    
    comp_counter = Counter()
    graph_sizes = []
    
    for fname in os.listdir(folder):
        if not fname.endswith(".gpickle"):
            continue
            
        with open(os.path.join(folder, fname), 'rb') as f:
            G = pickle.load(f)
        
        graph_sizes.append(G.number_of_nodes())
        
        for node, attr in G.nodes(data=True):
            if attr.get("type") in ["component"]:
                comp_type = attr.get("comp_type")
                comp_counter[comp_type] += 1
    
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


if __name__ == "__main__":
    print("Netlist parser running...")
    
    input_folder = "../netlists"
    output_folder = "../graphs_component_component"
    process_folder(input_folder, output_folder)
    remove_duplicate_graphs(output_folder)
    analyze_dataset(output_folder)