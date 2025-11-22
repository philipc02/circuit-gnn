import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from graph_descriptors import get_descriptor_dimension


class GNNEncoder(nn.Module):
    
    def __init__(self, hidden_channels, num_layers=3, gnn_type='gin', 
                 dropout=0.3, num_node_features=3):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # Input projection (from discrete features to embeddings)
        # Updated for 4-class problem (R, C, V, X)
        self.node_type_emb = nn.Embedding(5, hidden_channels)  # 4 types + padding
        self.comp_type_emb = nn.Embedding(6, hidden_channels)  # 4 types + padding + special
        self.pin_type_emb = nn.Embedding(6, hidden_channels)  # 4 pin types + padding + special
        
        # Initial projection
        self.input_proj = nn.Linear(hidden_channels * 3, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels * 2, hidden_channels)
                )
                conv = GINConv(mlp)
            elif gnn_type == 'gat':
                conv = GATConv(hidden_channels, hidden_channels // 4, heads=4, 
                              dropout=dropout, concat=True)
            elif gnn_type == 'gcn':
                conv = GCNConv(hidden_channels, hidden_channels)
            elif gnn_type == 'sage':
                conv = SAGEConv(hidden_channels, hidden_channels)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        # Embed discrete features
        node_type_idx = x[:, 0].clamp(min=0)
        comp_type_idx = x[:, 1].clamp(min=0)
        pin_type_idx = x[:, 2].clamp(min=0)
        
        node_emb = self.node_type_emb(node_type_idx)
        comp_emb = self.comp_type_emb(comp_type_idx)
        pin_emb = self.pin_type_emb(pin_type_idx)
        
        # Combine embeddings
        x = torch.cat([node_emb, comp_emb, pin_emb], dim=1)
        x = self.input_proj(x)
        
        # GNN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            
            # Residual connection (after first layer)
            if i > 0:
                x_new = x_new + x
            
            x = self.dropout(x_new)
        
        # Graph-level pooling
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_sum = global_add_pool(x, batch)
        
        # Combine different pooling strategies
        graph_embedding = torch.cat([graph_mean, graph_max, graph_sum], dim=1)
        
        return graph_embedding


class FEGIN(nn.Module):
    def __init__(self, hidden_channels, num_classes=4, num_layers=3, 
                 gnn_type='gin', dropout=0.3, n_eigenvalues=10,
                 dgsd_bins=10, use_dgsd=True, use_descriptors=True):
        super().__init__()
        
        self.use_descriptors = use_descriptors
        
        # GNN encoder
        self.gnn_encoder = GNNEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        # Graph descriptor dimension (with DGSD support)
        descriptor_dim = get_descriptor_dimension(
            n_eigenvalues, dgsd_bins
        ) if use_descriptors else 0

        # GNN produces 3 * hidden_channels (mean + max + sum pooling)
        gnn_output_dim = hidden_channels * 3
        
        # Descriptor MLP (if using descriptors)
        if use_descriptors:
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(descriptor_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            )
            fusion_input_dim = gnn_output_dim + hidden_channels
        else:
            self.descriptor_mlp = None
            fusion_input_dim = gnn_output_dim
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, data):
        # GNN encoding
        gnn_embedding = self.gnn_encoder(data.x, data.edge_index, data.batch)
        
        # Descriptor encoding (if available)
        if self.use_descriptors and hasattr(data, 'graph_descriptor'):
            descriptor_embedding = self.descriptor_mlp(data.graph_descriptor)
            # Concatenate GNN and descriptor embeddings
            combined = torch.cat([gnn_embedding, descriptor_embedding], dim=1)
        else:
            combined = gnn_embedding
        
        # Fusion and classification
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


class BaselineGNN(nn.Module):
    
    def __init__(self, hidden_channels, num_classes=8, num_layers=3,
                 gnn_type='gin', dropout=0.3):
        super().__init__()
        
        self.gnn_encoder = GNNEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        # Classifier
        gnn_output_dim = hidden_channels * 3
        self.classifier = nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, data):
        gnn_embedding = self.gnn_encoder(data.x, data.edge_index, data.batch)
        logits = self.classifier(gnn_embedding)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing FEGIN Model\n")
    
    # Create sample batch
    from torch_geometric.data import Data, Batch
    
    # Sample graph 1
    x1 = torch.randint(0, 8, (10, 3))  # 10 nodes, 3 features
    edge_index1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    descriptor1 = torch.randn(get_descriptor_dimension())
    data1 = Data(x=x1, edge_index=edge_index1, graph_descriptor=descriptor1)
    
    # Sample graph 2
    x2 = torch.randint(0, 8, (15, 3))
    edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    descriptor2 = torch.randn(get_descriptor_dimension())
    data2 = Data(x=x2, edge_index=edge_index2, graph_descriptor=descriptor2)
    
    # Create batch
    batch = Batch.from_data_list([data1, data2])
    
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
    
    # Test FEGIN
    print("FEGIN Model (with descriptors)")
    
    model_fegin = FEGIN(
        hidden_channels=128,
        num_classes=8,
        num_layers=3,
        gnn_type='gin',
        use_descriptors=True
    )
    
    print(f"Parameters: {count_parameters(model_fegin):,}")
    
    output = model_fegin(batch)
    print(f"Output shape: {output.shape}")
    print(f"Output (first graph): {output[0]}")
    
    # Test Baseline GNN
    print("Baseline GNN (without descriptors)")
    
    model_baseline = BaselineGNN(
        hidden_channels=128,
        num_classes=8,
        num_layers=3,
        gnn_type='gin'
    )
    
    print(f"Parameters: {count_parameters(model_baseline):,}")
    
    output = model_baseline(batch)
    print(f"Output shape: {output.shape}")
    
    # Test different GNN types
    print("Different GNN Types")
    
    for gnn_type in ['gin', 'gat', 'gcn', 'sage']:
        model = FEGIN(hidden_channels=64, gnn_type=gnn_type)
        output = model(batch)
        print(f"{gnn_type.upper():6s}: {count_parameters(model):>7,} params, output shape {output.shape}")
    
    print("FEGIN model ready for training!")
