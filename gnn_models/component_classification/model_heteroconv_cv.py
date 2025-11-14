import torch
import pickle
import os
from graph_to_heterodata_cv import graph_to_heterodata
from torch_geometric.data import Dataset
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GINConv, GCNConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import datetime
from pathlib import Path
import json
import pandas as pd
from torch_geometric.nn import global_mean_pool, global_max_pool
import numpy as np
import itertools

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

class CrossValidationDataset(Dataset):
    def __init__(self, fold_dir, split):
        self.folder = os.path.join(fold_dir, split)
        self.files = [f for f in os.listdir(self.folder) if f.endswith(".pt")]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return torch.load(os.path.join(self.folder, self.files[idx]))

class CVExperimentTracker:
    def __init__(self, experiment_name, fold_idx):
        self.experiment_name = experiment_name
        self.fold_idx = fold_idx
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path("cv_experiments") / f"{experiment_name}_fold{fold_idx}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [],
            'test_acc': None, 'test_f1': None, 'test_predictions': [], 'test_labels': []
        }
        self.config = {}

    def log_params(self, param_dict):
        self.param = param_dict
        with open(self.experiment_dir / "params.json", 'w') as f:
            json.dump(param_dict, f, indent=2)

    def log_metrics(self, epoch, train_loss, val_loss, val_acc, val_f1):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['val_f1'].append(val_f1)
        
        metrics_df = pd.DataFrame({
            'epoch': list(range(epoch + 1)),
            'train_loss': self.metrics['train_loss'],
            'val_loss': self.metrics['val_loss'], 
            'val_acc': self.metrics['val_acc'],
            'val_f1': self.metrics['val_f1']
        })
        metrics_df.to_csv(self.experiment_dir / "metrics.csv", index=False)

    def log_test_results(self, test_acc, test_f1, all_preds, all_labels):
        self.metrics['test_acc'] = test_acc
        self.metrics['test_f1'] = test_f1
        self.metrics['test_predictions'] = all_preds.tolist()
        self.metrics['test_labels'] = all_labels.tolist()
        
        with open(self.experiment_dir / "test_results.json", 'w') as f:
            json.dump({
                'test_acc': test_acc, 
                'test_f1': test_f1,
                'predictions': all_preds.tolist(),
                'labels': all_labels.tolist()
            }, f, indent=2)
        
        report = classification_report(all_labels, all_preds, output_dict=True)
        with open(self.experiment_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        cm = confusion_matrix(all_labels, all_preds)
        np.save(self.experiment_dir / "confusion_matrix.npy", cm)
        
        print(f"Fold {self.fold_idx} - Test accuracy: {test_acc:.4f} | Test f1: {test_f1:.4f}")

    def save_model(self, model, name="best_model.pth"):
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': self.param,
            'metrics': self.metrics,
            'fold_idx': self.fold_idx
        }, self.experiment_dir / name)

    def save_training_log(self, log_text):
        with open(self.experiment_dir / "training_log.txt", 'w') as f:
            f.write(log_text)

class RobustHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_layers=3, model_type='sage', 
                 dropout=0.3, use_residual=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.model_type = model_type
        self.use_residual = use_residual

        # Embedding layers
        self.node_type_emb = nn.Embedding(4, hidden_channels)
        self.comp_type_emb = nn.Embedding(9, hidden_channels)
        self.pin_type_emb = nn.Embedding(13, hidden_channels)

        # Convolution layers 
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if model_type == 'gin':
                conv_dict = {
                    ("component", "component_connection", "pin"): GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels)
                        )),
                    ("pin", "component_connection", "component"): GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels)
                        )),
                    ("subcircuit", "component_connection", "pin"): GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels)
                        )),
                    ("pin", "component_connection", "subcircuit"): GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels)
                        )),
                    ("pin", "net_connection", "net"): GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels)
                        )),
                    ("net", "net_connection", "pin"): GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels)
                        )),
                }
            elif model_type == 'gcn':
                conv_dict = {
                    ("component", "component_connection", "pin"): GCNConv(hidden_channels, hidden_channels),
                    ("pin", "component_connection", "component"): GCNConv(hidden_channels, hidden_channels),
                    ("subcircuit", "component_connection", "pin"): GCNConv(hidden_channels, hidden_channels),
                    ("pin", "component_connection", "subcircuit"): GCNConv(hidden_channels, hidden_channels),
                    ("pin", "net_connection", "net"): GCNConv(hidden_channels, hidden_channels),
                    ("net", "net_connection", "pin"): GCNConv(hidden_channels, hidden_channels),
                }
            else:  # SAGE 
                conv_dict = {
                    ("component", "component_connection", "pin"): SAGEConv(hidden_channels, hidden_channels),
                    ("pin", "component_connection", "component"): SAGEConv(hidden_channels, hidden_channels),
                    ("subcircuit", "component_connection", "pin"): SAGEConv(hidden_channels, hidden_channels),
                    ("pin", "component_connection", "subcircuit"): SAGEConv(hidden_channels, hidden_channels),
                    ("pin", "net_connection", "net"): SAGEConv(hidden_channels, hidden_channels),
                    ("net", "net_connection", "pin"): SAGEConv(hidden_channels, hidden_channels),
                }
            
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.dropout = nn.Dropout(dropout)
        
        # Classifier with flexible architecture
        classifier_layers = []
        in_features = hidden_channels * 2  # mean + max pool
        
        # First layer
        classifier_layers.extend([
            nn.Linear(in_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Optional second layer for larger networks
        if hidden_channels >= 128:
            classifier_layers.extend([
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7)  # Slightly less dropout
            ])
            final_in = hidden_channels // 2
        else:
            final_in = hidden_channels
            
        classifier_layers.append(nn.Linear(final_in, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Store initial embeddings for residual connections
        if self.use_residual:
            initial_embeddings = {}
            for node_type, x in x_dict.items():
                nt = x[:, 0]
                if node_type == "component":
                    ct = torch.zeros_like(nt)
                    pt = x[:, 2].clamp(min=0)
                else:
                    ct = x[:, 1].clamp(min=0)
                    pt = x[:, 2].clamp(min=0)

                x_emb = self.node_type_emb(nt) + self.comp_type_emb(ct) + self.pin_type_emb(pt)
                initial_embeddings[node_type] = x_emb
                x_dict[node_type] = x_emb
        else:
            for node_type, x in x_dict.items():
                nt = x[:, 0]
                if node_type == "component":
                    ct = torch.zeros_like(nt)
                    pt = x[:, 2].clamp(min=0)
                else:
                    ct = x[:, 1].clamp(min=0)
                    pt = x[:, 2].clamp(min=0)

                x_emb = self.node_type_emb(nt) + self.comp_type_emb(ct) + self.pin_type_emb(pt)
                x_dict[node_type] = x_emb

        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply residual connection if enabled
            if self.use_residual and i == 0:
                x_dict_new = {k: x + initial_embeddings.get(k, 0) for k, x in x_dict_new.items()}
            
            x_dict = {k: F.relu(x) for k, x in x_dict_new.items()}
            if i < len(self.convs) - 1:
                x_dict = {k: self.dropout(x) for k, x in x_dict.items()}

        component_embeddings = x_dict["component"]
        batch = data["component"].batch

        mean_pool = global_mean_pool(component_embeddings, batch)
        max_pool = global_max_pool(component_embeddings, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(graph_embedding)

def train_fold(fold_idx, config, base_data_folder="../../data/data_cross_validation_heterodata"):
    device = get_device()
    
    # Create experiment name from config
    exp_name_parts = [
        config['model_type'],
        f"h{config['hidden_channels']}",
        f"l{config['num_layers']}",
        f"d{config.get('dropout', 0.3)}",
        f"bs{config.get('batch_size', 32)}",
        f"lr{config.get('lr', 0.001)}",
    ]
    if config.get('use_residual', False):
        exp_name_parts.append("res")
    
    experiment_name = "_".join(exp_name_parts)
    tracker = CVExperimentTracker(experiment_name, fold_idx)
    
    params = {
        'fold_idx': fold_idx,
        'model_type': config['model_type'],
        'hidden_channels': config['hidden_channels'],
        'num_layers': config['num_layers'],
        'dropout': config.get('dropout', 0.3),
        'use_residual': config.get('use_residual', False),
        'learning_rate': config.get('lr', 0.001),
        'batch_size': config.get('batch_size', 32),
        'num_epochs': config.get('num_epochs', 100),
        'num_classes': 8,
        'device': str(device)
    }
    tracker.log_params(params)

    fold_dir = os.path.join(base_data_folder, f"fold_{fold_idx}")
    train_dataset = CrossValidationDataset(fold_dir, "train")
    val_dataset = CrossValidationDataset(fold_dir, "val")
    test_dataset = CrossValidationDataset(fold_dir, "test")
    
    print(f"Fold {fold_idx} - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32))

    model = RobustHeteroGNN(
        hidden_channels=config['hidden_channels'], 
        num_classes=8, 
        num_layers=config['num_layers'],
        model_type=config['model_type'],
        dropout=config.get('dropout', 0.3),
        use_residual=config.get('use_residual', False)
    ).to(device)
    
    print(f"Fold {fold_idx} - Training {config['model_type'].upper()} model")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: {config}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 0.001), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    patience = 20
    training_log = []

    for epoch in range(config.get('num_epochs', 100)):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            target = data.target_comp_type.to(device)
            
            out = model(data)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                target = data.target_comp_type.to(device)
                
                out = model(data)
                loss = criterion(out, target)
                val_loss += loss.item()

                preds = out.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(target.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        val_acc = (all_preds == all_labels).sum().item() / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        tracker.log_metrics(epoch, avg_train_loss, avg_val_loss, val_acc, val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            tracker.save_model(model, "best_model.pth")
            print(f"New best model saved with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        log_line = f"Fold {fold_idx} | Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}"
        if epoch % 10 == 0:
            print(log_line)
        training_log.append(log_line)

        if patience_counter >= patience:
            print(f"Fold {fold_idx} - Early stopping at epoch {epoch}")
            break

    # Test
    checkpoint = torch.load(tracker.experiment_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            target = data.target_comp_type.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_acc = (all_preds == all_labels).sum().item() / len(all_labels)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    tracker.log_test_results(test_acc, test_f1, all_preds, all_labels)
    tracker.save_training_log("\n".join(training_log))

    return test_acc, test_f1, all_preds, all_labels

def generate_hyperparameter_configs():
    base_configs = []
    
    # model types
    model_types = ['sage', 'gin', 'gcn']
    
    # smaller networks
    hidden_channels_options = [64, 128, 256]
    num_layers_options = [2, 3, 4]  # fewer layers
    
    # dropout rates
    dropout_options = [0.2, 0.3, 0.4]
    
    # batch sizes
    batch_size_options = [16, 32, 64]
    
    # learning rates
    lr_options = [0.001, 0.0005]
    
    # residual connections
    residual_options = [False, True]
    
    # different combinations
    for model_type in model_types:
        for hidden_channels in hidden_channels_options:
            for num_layers in num_layers_options:
                # skip too complex combinations for now as we have small graphs (expecting not much improvement)
                if hidden_channels >= 256 and num_layers >= 4:
                    continue
                    
                for dropout in dropout_options:
                    for batch_size in batch_size_options:
                        for lr in lr_options:
                            for use_residual in residual_options:
                                config = {
                                    'model_type': model_type,
                                    'hidden_channels': hidden_channels,
                                    'num_layers': num_layers,
                                    'dropout': dropout,
                                    'batch_size': batch_size,
                                    'lr': lr,
                                    'use_residual': use_residual,
                                    'num_epochs': 100
                                }
                                base_configs.append(config)
    
    # sort by model complexity (simpler first)
    base_configs.sort(key=lambda x: (x['hidden_channels'], x['num_layers']))
    
    return base_configs[:20]  # limit to top 20

def run_hyperparameter_search():
    configs = generate_hyperparameter_configs()
    
    print(f"Generated {len(configs)} hyperparam configs")
    print("Starting model training with generated hyperparam configs...")
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"Model: {config['model_type']}, Hidden: {config['hidden_channels']}, "
              f"Layers: {config['num_layers']}, Dropout: {config['dropout']}, "
              f"Residual: {config['use_residual']}, BS: {config['batch_size']}, LR: {config['lr']}")
        print(f"{'='*60}")
        
        fold_accuracies = []
        fold_f1_scores = []
        
        for fold_idx in range(5):
            print(f"\n--- Fold {fold_idx} ---")
            try:
                test_acc, test_f1, _, _ = train_fold(fold_idx, config)
                fold_accuracies.append(test_acc)
                fold_f1_scores.append(test_f1)
                print(f"Fold {fold_idx} completed: Acc={test_acc:.4f}, F1={test_f1:.4f}")
                
            except Exception as e:
                print(f"Fold {fold_idx} failed: {e}")
                fold_accuracies.append(0.0)
                fold_f1_scores.append(0.0)
        
        # Calculate cross-validation statistics
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        
        config_key = f"config_{i+1:02d}"
        results[config_key] = {
            'config': config,
            'fold_accuracies': fold_accuracies,
            'fold_f1_scores': fold_f1_scores,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_f1': mean_f1,
            'std_f1': std_f1
        }
        
        print(f"\nResults for {config_key}:")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")  # standard deviation
        print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    
    # Save results and find best configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("hyperparameter_search") / f"search_results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "hyperparameter_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best configuration
    best_config_key = max(results.keys(), key=lambda x: results[x]['mean_accuracy'])
    best_result = results[best_config_key]
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"BEST CONFIGURATION: {best_config_key}")
    print(f"Best Accuracy: {best_result['mean_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")
    print(f"Best F1-Score: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    print(f"Configuration: {best_result['config']}")
    
    # Save best configuration separately
    with open(results_dir / "best_configuration.json", 'w') as f:
        json.dump({
            'config_key': best_config_key,
            'config': best_result['config'],
            'mean_accuracy': best_result['mean_accuracy'],
            'std_accuracy': best_result['std_accuracy'],
            'mean_f1': best_result['mean_f1'],
            'std_f1': best_result['std_f1']
        }, f, indent=2)
    
    # Create results summary table
    summary_data = []
    for config_key, result in results.items():
        summary_data.append({
            'config_key': config_key,
            'model_type': result['config']['model_type'],
            'hidden_channels': result['config']['hidden_channels'],
            'num_layers': result['config']['num_layers'],
            'dropout': result['config']['dropout'],
            'use_residual': result['config']['use_residual'],
            'batch_size': result['config']['batch_size'],
            'lr': result['config']['lr'],
            'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'],
            'mean_f1': result['mean_f1'],
            'std_f1': result['std_f1']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('mean_accuracy', ascending=False)
    summary_df.to_csv(results_dir / "results_summary.csv", index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Top 5 configurations:")
    print(summary_df.head(5)[['config_key', 'model_type', 'hidden_channels', 'num_layers', 'mean_accuracy', 'std_accuracy']])
    
    return results, best_result

if __name__ == "__main__":
    print("Starting Hyperparameter Search...")
    results, best_result = run_hyperparameter_search()