import torch
import pickle
import os
from graph_to_heterodata import graph_to_heterodata
from torch_geometric.data import Dataset
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import datetime
from pathlib import Path
import json
import pandas as pd

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

class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path("experiments") / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [],
            'test_acc': None, 'test_f1': None
        }
        self.config = {}

    def log_params(self, param_dict):
        # log experiment parameters
        self.param = param_dict
        with open(self.experiment_dir / "param.json", 'w') as f:
            json.dump(param_dict, f, indent=2)

    def log_metrics(self, epoch, train_loss, val_loss, val_acc, val_f1):
        # log metrics for each epoch
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['val_f1'].append(val_f1)
        
        # save metrics to CSV after each epoch
        metrics_df = pd.DataFrame({
            'epoch': list(range(epoch + 1)),
            'train_loss': self.metrics['train_loss'],
            'val_loss': self.metrics['val_loss'], 
            'val_acc': self.metrics['val_acc'],
            'val_f1': self.metrics['val_f1']
        })
        metrics_df.to_csv(self.experiment_dir / "metrics.csv", index=False)

    def log_test_results(self, test_acc, test_f1):
        # log final test results
        self.metrics['test_acc'] = test_acc
        self.metrics['test_f1'] = test_f1
        with open(self.experiment_dir / "test_results.json", 'w') as f:
            json.dump({'test_acc': test_acc, 'test_f1': test_f1}, f, indent=2)

    def save_model(self, model, name="best_model.pth"):
        torch.save({
            'model_state_dict': model.state_dict(),
            'param': self.param,
            'metrics': self.metrics
        }, self.experiment_dir / name)

    def save_training_log(self, log_text):
        # save training log text
        with open(self.experiment_dir / "training_log.txt", 'w') as f:
            f.write(log_text)

# class definition for HeteroConv model with GraphSAGE layers
class HeteroSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_layers=3):
        super().__init__()

        # add embedding layers using nn.Embedding
        self.node_type_emb = nn.Embedding(4, hidden_channels)      # component, pin, net, subcircuit
        self.comp_type_emb = nn.Embedding(9, hidden_channels)     # R, C, L, V, M, Q, D, I and one if no component type
        self.pin_type_emb  = nn.Embedding(13, hidden_channels)     # 1, 2, pos, neg, drain, gate, source, collector, base, emitter, anode, cathode and one if no pin type

        # layer 1: convert input embeddings to hidden dimension and if multiple relations point to same destination, results will be aggregated using sum
        # different message passing rules for different edge types
        # -1: input shape inferred automatically (tuple as we have (in_src, in_dst) hetero graphs could have different feature sizes for diff node types (not in this case))
        # outpit dim is hidden_channels
        # for all layers after first:
        # input shape here is known already from prev layer
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {
                ("component", "component_connection", "pin"): SAGEConv((-1, -1) if i == 0 else (hidden_channels, hidden_channels), hidden_channels),
                ("pin", "component_connection", "component"): SAGEConv((-1, -1) if i == 0 else (hidden_channels, hidden_channels), hidden_channels),
                ("subcircuit", "component_connection", "pin"): SAGEConv((-1, -1) if i == 0 else (hidden_channels, hidden_channels), hidden_channels),
                ("pin", "component_connection", "subcircuit"): SAGEConv((-1, -1) if i == 0 else (hidden_channels, hidden_channels), hidden_channels),
                ("pin", "net_connection", "net"): SAGEConv((-1, -1) if i == 0 else (hidden_channels, hidden_channels), hidden_channels),
                ("net", "net_connection", "pin"): SAGEConv((-1, -1) if i == 0 else (hidden_channels, hidden_channels), hidden_channels),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # add dropout for regularization (preventing overfitting, better generalization): randomly disable neurons during training with probability p 
        # -> forces remaining neurons to learn more robust features
        # here 30 percent of neurons are disabled during training
        self.dropout = nn.Dropout(0.3)

        # classifier head only for component nodes (the ones with labels)
        # self.classifier = nn.Linear(hidden_channels, num_classes)

        # non linear classifier to capture more complex patterns (multi layer)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),   # layer 1
            nn.ReLU(),  # activation
            nn.Dropout(0.2),    # regularization
            nn.Linear(hidden_channels // 2, num_classes)    # layer 2
        )

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

        # message passing for all layers with dropout
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # ReLU (Rectified Linear Unit): activation function between layers in NN -> ReLU(x) = max(0, x) (if value is positive, keep, else set to zero) => helps network learn nonlinear patterns
            x_dict = {k: x.relu() for k, x in x_dict.items()}
            if i < len(self.convs) - 1:  # no dropout after last layer
                x_dict = {k: self.dropout(x) for k, x in x_dict.items()}

        # return logits for component nodes only
        # classifying into 9 component types (subcircuit excluded), the model needs to output 9 logits (raw, unnormalized output values from the final layer of a model, just before applying f.ex. softmax) per node
        return self.classifier(x_dict["component"])
    
def train_model(num_epochs=50, hidden_channels=64, lr=0.001, batch_size=1, num_layers=3):

    tracker = ExperimentTracker("hetero_component_classification")
    
    # log param configuration
    params = {
        'num_epochs': num_epochs,
        'hidden_channels': hidden_channels,
        'learning_rate': lr,
        'batch_size': batch_size,
        'num_layers': num_layers,
        'num_classes': 9,
        'weight_decay': 1e-5
    }
    tracker.log_params(params)

    train_dataset, val_dataset, test_dataset = load_dataset()
    # loads graphs in batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle the order of samples at every epoch to avoid pattrern learning
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_classes = 9 # nodes can have following component_type values: (R, C, L, V, M, Q, D, I -> corresponding index) or no component type (-1)

    model = HeteroSAGE(hidden_channels=hidden_channels, num_classes=num_classes, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Adam -> gradient descent optimizer for model's weights (uses computed gradients)
    criterion = torch.nn.CrossEntropyLoss() # compute loss using cross entropy between predicted and desired output

    # training variables
    best_val_acc = 0
    patience_counter = 0
    patience = 20   # if patience counter surpasses this value we stop training as we dont see any improvement in val acc even after this many epochs
    training_log = []

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

        # log metrics for each epoch
        tracker.log_metrics(epoch, avg_train_loss, avg_val_loss, val_acc, val_f1)

        # save the model with highest score currently
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0    # reset counter
            tracker.save_model(model, "best_model.pth")
            print(f"New best model saved with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        log_line = f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        print(log_line)
        training_log.append(log_line)

        # stop training as we dont see any improvement in val acc even after 20 epochs
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # load best model for testing
    checkpoint = torch.load(tracker.experiment_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

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
    print(f"Test accuracy: {test_acc:.4f} | Test f1: {test_f1:.4f}")

    # log test results and save taining log (turn list into string first)
    tracker.log_test_results(test_acc, test_f1)
    tracker.save_training_log("\n".join(training_log))


    return model

if __name__ == "__main__":
    # create_heterodata_obj()
    train_model()