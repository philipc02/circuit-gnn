import os
import networkx as nx
import pickle
from collections import defaultdict


def star_graph_to_component_level(G_star: nx.Graph) -> nx.Graph:
    # Convert star graph to component level representation so we have baseline comparison to original circuit data representation
    # Star graph: Component -> Pin -> Net -> Pin -> Component
    # Component level: Component -> Net -> Component

    G_comp = nx.Graph()
    
    # add all component nodes with their features
    component_nodes = []
    for node, attr in G_star.nodes(data=True):
        if attr.get("type") == "component":
            G_comp.add_node(node, **attr)
            component_nodes.append(node)
        elif attr.get("type") == "subcircuit":
            G_comp.add_node(node, **attr)
            component_nodes.append(node)

    # add net nodes (keep them as nodes!)
    net_nodes = []
    for node, attr in G_star.nodes(data=True):
        if attr.get("type") == "net":
            # Add net as a node with a special type
            G_comp.add_node(node, 
                              type="net",
                              net_name=attr.get("net_name", "unknown"))
            net_nodes.append(node)
    

    # Component -> Pin -> Net becomes Component -> Net
    for node, attr in G_star.nodes(data=True):
        if attr.get("type") == "pin":
            component = attr.get("component")
            if component is None:
                continue
            
            # Find nets this pin connects to
            for neighbor in G_star.neighbors(node):
                net_attr = G_star.nodes[neighbor]
                if net_attr.get("type") == "net":
                    G_comp.add_edge(component, neighbor)
    
    # Store metadata
    G_comp.graph['representation'] = 'component_net_level'
    G_comp.graph['num_components'] = len(component_nodes)
    G_comp.graph['num_nets'] = len(net_nodes)
    G_comp.graph['original_star_graph'] = G_star.graph.get('original_id', 'unknown')
    
    return G_comp


def process_all_folds_to_component_level(
    input_base_folder="../../data/data_kfold_filtered",
    output_base_folder="../../data/data_kfold_filtered_component_level",
    num_folds=5
):

    
    for fold_idx in range(num_folds):
        print(f"Processing Fold {fold_idx}")
        
        fold_input = os.path.join(input_base_folder, f"fold_{fold_idx}")
        fold_output = os.path.join(output_base_folder, f"fold_{fold_idx}")
        
        for split in ['train', 'val', 'test']:
            split_input = os.path.join(fold_input, split)
            split_output = os.path.join(fold_output, split)
            os.makedirs(split_output, exist_ok=True)
            
            if not os.path.exists(split_input):
                print(f"Warning: {split_input} does not exist, skipping...")
                continue
            
            files = [f for f in os.listdir(split_input) if f.endswith('.gpickle')]
            print(f"\n{split.upper()}: Converting {len(files)} graphs...")
            
            converted = 0
            skipped = 0
            
            for fname in files:
                try:
                    # load star graph
                    with open(os.path.join(split_input, fname), 'rb') as f:
                        G_star = pickle.load(f)
                    
                    # Convert to component level
                    G_comp = star_graph_to_component_level(G_star)
                    
                    # shouldn't happen
                    if G_comp.graph.get('num_components', 0) == 0:
                        print(f"  Skipping {fname}: no components")
                        skipped += 1
                        continue
                    
                    # Save with same filename for easy tracking
                    output_path = os.path.join(split_output, fname)
                    with open(output_path, 'wb') as f:
                        pickle.dump(G_comp, f)
                    
                    converted += 1
                    
                    if converted % 50 == 0:
                        print(f"  Converted {converted}/{len(files)}...")
                        
                except Exception as e:
                    print(f"  Error converting {fname}: {e}")
                    skipped += 1
            
            print(f"  Completed: {converted} converted, {skipped} skipped")
    
    print("Component-level baseline dataset created!")
    print(f"Location: {output_base_folder}")


def verify_conversion(star_graph_path, comp_net_graph_path):
    """Verify conversion preserved information."""
    with open(star_graph_path, 'rb') as f:
        G_star = pickle.load(f)
    
    with open(comp_net_graph_path, 'rb') as f:
        G_comp_net = pickle.load(f)
    
    star_components = sum(1 for _, attr in G_star.nodes(data=True) 
                         if attr.get("type") in ["component", "subcircuit"])
    star_nets = sum(1 for _, attr in G_star.nodes(data=True) 
                   if attr.get("type") == "net")
    
    comp_net_components = sum(1 for _, attr in G_comp_net.nodes(data=True)
                             if attr.get("type") in ["component", "subcircuit"])
    comp_net_nets = sum(1 for _, attr in G_comp_net.nodes(data=True)
                       if attr.get("type") == "net")
    
    print(f"\nVerification:")
    print(f"  Star graph - Components: {star_components}, Nets: {star_nets}")
    print(f"  Baseline graph - Components: {comp_net_components}, Nets: {comp_net_nets}")
    print(f"  Components match: {star_components == comp_net_components}")
    print(f"  Nets match: {star_nets == comp_net_nets}")
    print(f"  Total nodes - Star: {G_star.number_of_nodes()}, Baseline: {G_comp_net.number_of_nodes()}")
    
    return (star_components == comp_net_components and 
            star_nets == comp_net_nets)



if __name__ == "__main__":
    print("Creating Component-Level Baseline Dataset")
    process_all_folds_to_component_level()

    print("Verifying Conversion (Fold 0, Train, First File)")
    
    star_folder = "../../data/data_kfold_filtered/fold_0/train"
    comp_folder = "../../data/data_kfold_filtered_component_level/fold_0/train"
    
    if os.path.exists(star_folder) and os.path.exists(comp_folder):
        star_files = [f for f in os.listdir(star_folder) if f.endswith('.gpickle')]
        if star_files:
            test_file = star_files[0]
            verify_conversion(
                os.path.join(star_folder, test_file),
                os.path.join(comp_folder, test_file)
            )
    
    print("Component-level baseline ready for FEGIN")
