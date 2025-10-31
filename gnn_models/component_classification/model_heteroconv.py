import torch
import pickle
import os
from graph_to_heterodata import graph_to_heterodata
from torch_geometric.data import Dataset
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

def create_heterodata_obj():
    input_folder = "../../data/data_conventional"
    output_folder = "../../data/data_conventional_heterodata"

    splits = ["train", "val", "test"]

    for split in splits:
        in_dir = os.path.join(input_folder, split)
        out_dir = os.path.join(output_folder, split)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(in_dir):
            if fname.endswith(".gpickle"):
                with open(os.path.join(in_dir, fname), "rb") as f:
                    G = pickle.load(f)

                data = graph_to_heterodata(G)
                # save heterodata object into output_folder as .pt object
                torch.save(data, os.path.join(out_dir, fname.replace(".gpickle", ".pt")))

    print("HeteroData objects were created for all graphs.\n")

def load_dataset():
    # define dataset class
    class HeteroCircuitDataset(Dataset):
        def __init__(self, folder):
            self.folder = folder
            self.files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            return torch.load(os.path.join(self.folder, self.files[idx]))

    return (
        HeteroCircuitDataset("../../data/data_conventional_heterodata/train"),
        HeteroCircuitDataset("../../data/data_conventional_heterodata/val"),
        HeteroCircuitDataset("../../data/data_conventional_heterodata/test"),
    )

# class definition for HeteroConv model with GraphSAGE layers
class HeteroSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()

        # add embedding layers using nn.Embedding
        self.node_type_emb = nn.Embedding(4, hidden_channels)      # component, pin, net, subcircuit
        self.comp_type_emb = nn.Embedding(9, hidden_channels)     # R, C, L, V, M, Q, D, I and one if no component type
        self.pin_type_emb  = nn.Embedding(13, hidden_channels)     # 1, 2, pos, neg, drain, gate, source, collector, base, emitter, anode, cathode and one if no pin type

        # layer 1: convert input embeddings to hidden dimension and if multiple relations point to same destination, results will be aggregated using sum
        # different message passing rules for different edge types
        # -1: input shape inferred automatically (tuple as we have (in_src, in_dst) hetero graphs could have different feature sizes for diff node types (not in this case))
        # outpit dim is hidden_channels
        self.convs1 = HeteroConv({
            ("component", "component_connection", "pin"): SAGEConv((-1, -1), hidden_channels),
            ("pin", "component_connection", "component"): SAGEConv((-1, -1), hidden_channels),
            ("subcircuit", "component_connection", "pin"): SAGEConv((-1, -1), hidden_channels),
            ("pin", "component_connection", "subcircuit"): SAGEConv((-1, -1), hidden_channels),
            ("pin", "net_connection", "net"): SAGEConv((-1, -1), hidden_channels),
            ("net", "net_connection", "pin"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        # layer 2: second propagation
        # input shape here is known already from prev layer
        self.convs2 = HeteroConv({
            ("component", "component_connection", "pin"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("pin", "component_connection", "component"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("subcircuit", "component_connection", "pin"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("pin", "component_connection", "subcircuit"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("pin", "net_connection", "net"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("net", "net_connection", "pin"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
        }, aggr="sum")

        # classifier head only for component nodes (the ones with labels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    # forward pass
    def forward(self, x_dict, edge_index_dict): # takes node features and graph structure; x_dict and edge_index_dict are needed as they store the features for the different node and edge types
        
        # turn the indices into embeddings
        for node_type, x in x_dict.items():
            # x has shape [num_nodes, 3] -> for each node (node_type_idx, comp_type_idx, pin_type_idx)
            nt = x[:, 0]

            if node_type == "component":
                # don't add comp_type to embedding, model should predict this uisng the other features
                ct = torch.zeros_like(nt)  # zero instead of actual comp_type
                pt = x[:, 2].clamp(min=0) # clamps tensor elements to start from min val: -1 ais 0, rest starts from 1
            else:
                ct = x[:, 1].clamp(min=0)
                pt = x[:, 2].clamp(min=0)

            x_emb = (self.node_type_emb(nt) + self.comp_type_emb(ct) + self.pin_type_emb(pt))

            x_dict[node_type] = x_emb # replace tuples with indices with new embedding

        # first message passing
        x_dict = self.convs1(x_dict, edge_index_dict)
        # ReLU (Rectified Linear Unit): activation function between layers in NN -> ReLU(x) = max(0, x) (if value is positive, keep, else set to zero) => helps network learn nonlinear patterns
        x_dict = {k: x.relu() for k, x in x_dict.items()}

        # second message passing
        x_dict = self.convs2(x_dict, edge_index_dict)
        # relu activation
        x_dict = {k: x.relu() for k, x in x_dict.items()}

        # return logits for component nodes only
        # classifying into 8 component types (subcircuit excluded), the model needs to output 8 logits (raw, unnormalized output values from the final layer of a model, just before applying f.ex. softmax) per node
        return self.classifier(x_dict["component"])
    
def train_model(num_epochs=50, hidden_channels=64, lr=0.001, batch_size=1):
    train_dataset, val_dataset, test_dataset = load_dataset()
    # loads graphs in batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle the order of samples at every epoch to avoid pattrern learning
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_classes = 9 # nodes can have following component_type values: (R, C, L, V, M, Q, D, I -> corresponding index) or no component type (-1)

    model = HeteroSAGE(hidden_channels=hidden_channels, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Adam -> gradient descent optimizer for model's weights (uses computed gradients)
    criterion = torch.nn.CrossEntropyLoss() # compute loss using cross entropy between predicted and desired output

    for epoch in range(num_epochs):
        ## train
        model.train()
        total_loss = 0

        # compute loss for each graph in train dataset and add up
        for data in train_loader:
            out = model(data.x_dict, data.edge_index_dict) # forward pass: predict from data (using HeteroSAGE forward() function defined above)
            loss = criterion(out, data["component"].y)

            optimizer.zero_grad() # resets gradients from previous iteration (otherwise gradients from multiple backward passes will add up -> like this only the current gradient is used to update the weights)
            loss.backward() # compute gradients (backward pass)
            optimizer.step()  # update weights (model parameters) using gradients
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        ## validation to monitor overfitting (implement early stopping if train loss and val loss shows big gap)
        model.eval()
        val_loss = 0
        # to calculate validation accuracy
        all_preds = []
        all_labels = []
        with torch.no_grad(): # tells PyTorch not to track gradients (only needed during training when we are trying to update weights)
            for data in val_loader:
                out = model(data.x_dict, data.edge_index_dict)
                loss = criterion(out, data["component"].y)
                val_loss += loss.item()

                preds = out.argmax(dim=1)   # predicted component types
                all_preds.append(preds)
                all_labels.append(data["component"].y)  # actual component type

        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        # percentage of correct predictions
        val_acc = (all_preds == all_labels).sum().item() / len(all_labels)
        # f1 score accounts for class imbalances, micro score gives more weight to majority classes (performance on rare occurance component types is not as important)
        val_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='micro')

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    ## test
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            out = model(data.x_dict, data.edge_index_dict)
            preds = out.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(data["component"].y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_acc = (all_preds == all_labels).sum().item() / len(all_labels)
    test_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='micro')
    print(f"Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    return model

if __name__ == "__main__":
    # create_heterodata_obj()
    train_model()