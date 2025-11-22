import torch
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_layers=3):
        super().__init__()
        self.node_type_emb = nn.Embedding(4, hidden_channels)
        self.comp_type_emb = nn.Embedding(9, hidden_channels)
        self.pin_type_emb = nn.Embedding(13, hidden_channels)

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

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            nt = x[:, 0]
            if node_type == "component":
                ct = torch.zeros_like(nt)
                pt = x[:, 2].clamp(min=0)
            else:
                ct = x[:, 1].clamp(min=0)
                pt = x[:, 2].clamp(min=0)

            x_emb = (self.node_type_emb(nt) + self.comp_type_emb(ct) + self.pin_type_emb(pt))
            x_dict[node_type] = x_emb

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: x.relu() for k, x in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {k: self.dropout(x) for k, x in x_dict.items()}

        return self.classifier(x_dict["component"])

def load_model(model_path):
    # load the trained model from checkpoint
    checkpoint = torch.load(model_path)
    
    # get parameters
    params = checkpoint['param']
    
    # initialize model with same architecture
    model = HeteroSAGE(hidden_channels=params['hidden_channels'], num_classes=params['num_classes'], num_layers=params['num_layers'])
    
    # load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    # set to evaluation mode
    model.eval()
    
    return model

def predict_one_circuit(model, circuit_data):
    # run inference on one circuit
    with torch.no_grad():
        logits = model(circuit_data.x_dict, circuit_data.edge_index_dict)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
    
class HeteroCircuitDataset(Dataset):
        def __init__(self, folder):
            self.folder = folder
            self.files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            file_name = self.files[idx]
            data = torch.load(os.path.join(self.folder, file_name))
            data.file_name = file_name
            return data

def run_inference():
    # inference for manual inspection
    
    model_path = "experiments/hetero_component_classification_20251101_115134/best_model.pth"
    model = load_model(model_path)
    
    # load test dataset
    test_dataset = HeteroCircuitDataset("../../data/data_conventional_heterodata/test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    comp_type_mapping = {0: 'R', 1: 'C', 2: 'L', 3: 'V', 4: 'M', 5: 'Q', 6: 'D', 7: 'I', 8: 'None'}
    
    print("Running inference...\n")
        
    for i, data in enumerate(test_loader):
        file_name = data.file_name[0] if hasattr(data, 'file_name') else f"unknown_{i}"
        result = predict_one_circuit(model, data)
        
        true_labels = data["component"].y.numpy()  # pytorch tensor convert to numpy array
        predictions = result['predictions']
        probabilities = result['probabilities']
        
        correct = (predictions == true_labels).sum()
        total = len(true_labels)
        accuracy = correct / total
        
        print(f"\nCircuit {i+1}: {file_name}\n")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct)\n")
        
        # some individual predictions
        print("Sample predictions:\n")
        for j in range(min(5, len(predictions))):  # first 5 predictions
            true_comp = comp_type_mapping.get(true_labels[j], 'None')
            pred_comp = comp_type_mapping.get(predictions[j], 'None')
            confidence = probabilities[j][predictions[j]]
            
            print(f"Node {j}: Actual = {true_comp}, Predicted = {pred_comp}, Confidence = {confidence:.3f}")

if __name__ == "__main__":
    run_inference()