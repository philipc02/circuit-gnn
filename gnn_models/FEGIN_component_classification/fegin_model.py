# FIXED FEGIN Model with support for component-level representation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, global_sort_pool
from graph_descriptors import get_descriptor_dimension


class GNNEncoder(nn.Module):
    
    def __init__(self, hidden_channels, num_layers=3, gnn_type='gin', 
                 dropout=0.3, num_node_features=3):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # UPDATED: Embedding dimensions to include special tokens for component representation
        self.node_type_emb = nn.Embedding(6, hidden_channels)  # 4 types + padding + masked
        self.comp_type_emb = nn.Embedding(8, hidden_channels)  # 4 types + padding + special + masked + context
        self.pin_type_emb = nn.Embedding(7, hidden_channels)   # 4 pin types + padding + special + masked
        
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
    
    def forward(self, x, edge_index, batch, component_mask=None):
        """
        Forward pass with optional component masking for component representation
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            component_mask: Optional mask indicating which nodes are components (for component repr)
        """
        x_embedded = self.get_node_embeddings(x, edge_index)
        
        # If component mask provided, use weighted pooling that focuses on components
        if component_mask is not None:
            # Use component-aware pooling
            graph_mean = self.component_aware_pool(x_embedded, batch, component_mask, 'mean')
            graph_max = self.component_aware_pool(x_embedded, batch, component_mask, 'max')
            graph_sum = self.component_aware_pool(x_embedded, batch, component_mask, 'sum')
        else:
            # Standard pooling
            graph_mean = global_mean_pool(x_embedded, batch)
            graph_max = global_max_pool(x_embedded, batch)
            graph_sum = global_add_pool(x_embedded, batch)
        
        # Combine different pooling strategies
        graph_embedding = torch.cat([graph_mean, graph_max, graph_sum], dim=1)

        return graph_embedding
    
    def component_aware_pool(self, x, batch, component_mask, pool_type='mean'):
        """
        Pooling that gives more weight to component nodes
        
        This helps the model focus on component-level information
        while still utilizing structural context from pins/nets
        """
        # Weight component nodes more heavily (3x)
        weights = component_mask.float() * 2.0 + 1.0  # components: 3.0, others: 1.0
        weights = weights.unsqueeze(1)  # [num_nodes, 1]
        
        x_weighted = x * weights
        
        if pool_type == 'mean':
            # Weighted mean pooling
            sum_weighted = global_add_pool(x_weighted, batch)
            sum_weights = global_add_pool(weights, batch)
            return sum_weighted / (sum_weights + 1e-8)
        elif pool_type == 'max':
            return global_max_pool(x_weighted, batch)
        elif pool_type == 'sum':
            return global_add_pool(x_weighted, batch)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

    def get_node_embeddings(self, x, edge_index):
        # Embed discrete features
        node_type_idx = x[:, 0].clamp(min=0, max=5)
        comp_type_idx = x[:, 1].clamp(min=0, max=7)  # Updated max for context token
        pin_type_idx = x[:, 2].clamp(min=0, max=6)
        
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
        return x


class FEGIN(nn.Module):
    def __init__(self, hidden_channels, num_classes=4, num_layers=3, 
                 gnn_type='gin', dropout=0.3, n_eigenvalues=10,
                 dgsd_bins=10, use_dgsd=True, use_descriptors=True, k=30):
        super().__init__()
        
        self.use_descriptors = use_descriptors
        self.k = k
        
        # GNN encoder
        self.gnn_encoder = GNNEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        # Graph descriptor dimension
        descriptor_dim = get_descriptor_dimension(
            n_eigenvalues, dgsd_bins
        ) if use_descriptors else 0

        sort_pool_dim = k * hidden_channels
        traditional_gnn_output_dim = hidden_channels * 3
        
        # Descriptor MLP
        if use_descriptors:
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(descriptor_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            )
            fusion_input_dim = traditional_gnn_output_dim + sort_pool_dim + hidden_channels
        else:
            self.descriptor_mlp = None
            fusion_input_dim = traditional_gnn_output_dim + sort_pool_dim
        
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
        # Check if we have component mask (for component representation)
        component_mask = data.is_component if hasattr(data, 'is_component') else None
        
        # GNN encoding with component-aware pooling if mask available
        gnn_embedding = self.gnn_encoder.get_node_embeddings(data.x, data.edge_index)

        traditional_pool = self.gnn_encoder(
            data.x, data.edge_index, data.batch, component_mask
        )
        sort_pooling_embedding = global_sort_pool(gnn_embedding, data.batch, k=self.k)

        # Combine pooling strategies
        gnn_combined = torch.cat([traditional_pool, sort_pooling_embedding], dim=1)
        
        # Descriptor encoding
        if self.use_descriptors and hasattr(data, 'graph_descriptor'):
            descriptor_embedding = self.descriptor_mlp(data.graph_descriptor)
            combined = torch.cat([gnn_combined, descriptor_embedding], dim=1)
        else:
            combined = gnn_combined
        
        # Fusion and classification
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


class BaselineGNN(nn.Module):
    """Baseline without descriptors"""
    
    def __init__(self, hidden_channels, num_classes=4, num_layers=3,
                 gnn_type='gin', dropout=0.3):
        super().__init__()
        
        self.gnn_encoder = GNNEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
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
        component_mask = data.is_component if hasattr(data, 'is_component') else None
        gnn_embedding = self.gnn_encoder(
            data.x, data.edge_index, data.batch, component_mask
        )
        logits = self.classifier(gnn_embedding)
        return logits


if __name__ == "__main__":
    print("Testing FIXED FEGIN Model\n")
    
    from torch_geometric.data import Data, Batch
    
    # Test with component mask
    x = torch.randint(0, 4, (10, 3))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    descriptor = torch.randn(get_descriptor_dimension())
    is_component = torch.tensor([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])  # 4 components, 6 context nodes
    
    data = Data(
        x=x, 
        edge_index=edge_index, 
        graph_descriptor=descriptor,
        is_component=is_component
    )
    
    batch = Batch.from_data_list([data])
    
    print("Testing component-aware model:")
    model = FEGIN(hidden_channels=64, num_classes=4, use_descriptors=True)
    
    output = model(batch)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✓ Model successfully handles component representation!")