# Adjusted training script for FEGIN component classification with cross validation

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
print("Using DataLoader:", DataLoader)
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
import datetime

from fegin_model import FEGIN, BaselineGNN
from fegin_dataset import FEGINDatasetFiltered, collate_fegin

COMPONENT_TYPES = ["R", "C", "V", "X"]


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def compute_class_weights(dataset, device):
    # imbalanced dataset!
    '''
    From earlier dataset analysis:
    R    :   845 ( 45.5%)
    C    :   706 ( 38.0%)
    V    :   305 ( 16.4%)
    X    :   248 ( 11.8%)
    '''
    # Counts from analysis
    counts = np.array([845, 706, 305, 248])
    
    # Compute weights (inverse of frequency)
    weights = 1.0 / (counts / counts.sum())
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, criterion, device):
    # Train one epoch
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for data in loader:
        if data is None:
            continue
        
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out, data.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, return_details=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for data in loader:
            if data is None:
                continue
            
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            total_samples += data.num_graphs
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    if len(all_preds) == 0:
        return avg_loss, 0, 0, None, None
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    if return_details:
        return avg_loss, accuracy, f1_weighted, f1_macro, all_preds, all_labels
    
    return avg_loss, accuracy, f1_weighted, f1_macro


def compute_per_class_metrics(labels, preds):
    # per-class accuracy and F1
    metrics = {}
    
    for i, comp_type in enumerate(COMPONENT_TYPES):
        mask = labels == i
        if mask.sum() == 0:
            metrics[comp_type] = {'accuracy': 0, 'f1': 0, 'support': 0}
            continue
        
        class_acc = (preds[mask] == labels[mask]).mean()
        class_f1 = f1_score(labels == i, preds == i, zero_division = 0)
        
        metrics[comp_type] = {
            'accuracy': float(class_acc),
            'f1': float(class_f1),
            'support': int(mask.sum()) # TODO
        }
    
    return metrics


def train_fold(fold_idx, config, representation='star',
               base_data_folder_star="../../data/data_kfold_filtered",
               base_data_folder_comp="../../data/data_kfold_filtered_component_level"):

    # train FEGIN for one fold
    device = get_device()
    
    # select data folder based on representation
    if representation == 'star':
        base_folder = base_data_folder_star
    elif representation == 'component':
        base_folder = base_data_folder_comp
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    fold_dir = f"{base_folder}/fold_{fold_idx}"
    
    print(f"Fold {fold_idx} | Representation: {representation}")
    
    # Create datasets
    train_dataset = FEGINDatasetFiltered(
        fold_dir, 'train', 
        representation=representation,
        mask_strategy='keep_pins'
    )
    
    val_dataset = FEGINDatasetFiltered(
        fold_dir, 'val',
        representation=representation,
        mask_strategy='keep_pins'
    )
    
    test_dataset = FEGINDatasetFiltered(
        fold_dir, 'test',
        representation=representation,
        mask_strategy='keep_pins'
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Debug: Check descriptor dimensions across multiple samples
    print("\n=== Descriptor Dimension Debug ===")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        if sample is not None and hasattr(sample, 'graph_descriptor'):
            print(f"Sample {i}: descriptor shape = {sample.graph_descriptor.shape}")
            print(f"Sample {i}: descriptor dim = {sample.graph_descriptor.shape[0]}")
    print("=== End Debug ===\n")
    
    # class weights
    class_weights = compute_class_weights(train_dataset, device)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fegin
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fegin
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fegin
    )
    
    # Create model
    if config.get('use_descriptors', True):
        sample_data = train_dataset[0]
        if sample_data is not None and hasattr(sample_data, 'graph_descriptor'):
            # Get the actual descriptor dimension from the computed descriptors
            actual_descriptor_dim = int(sample_data.graph_descriptor.shape[0])
            print(f"Actual descriptor dimension: {actual_descriptor_dim}")
            
            # Debug: Check what the descriptor looks like
            print("Raw descriptor from dataset:", sample_data.graph_descriptor.shape)
            print("Descriptor values:", sample_data.graph_descriptor)
        else:
            # Use the expected dimension from graph_descriptors.py
            from graph_descriptors import get_descriptor_dimension
            actual_descriptor_dim = get_descriptor_dimension()
            print(f"Using expected descriptor dimension: {actual_descriptor_dim}")
        print("Raw descriptor from dataset:", train_dataset[0].graph_descriptor.shape)
        print(train_dataset[0].graph_descriptor)
        model = FEGIN(
            hidden_channels=config['hidden_channels'],
            num_classes=len(COMPONENT_TYPES),
            num_layers=config['num_layers'],
            gnn_type=config['gnn_type'],
            dropout=config['dropout'],
            use_descriptors=True,
            descriptor_dim=actual_descriptor_dim
        ).to(device)
    else:
        model = BaselineGNN(
            hidden_channels=config['hidden_channels'],
            num_classes=len(COMPONENT_TYPES),
            num_layers=config['num_layers'],
            gnn_type=config['gnn_type'],
            dropout=config['dropout']
        ).to(device)
    
    print(f"Model: {config['gnn_type'].upper()} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_val_f1 = 0
    best_val_acc = 0
    patience_counter = 0
    patience = config.get('patience', 30)
    
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1_weighted, val_f1_macro = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_f1_weighted) # adjust model based on weighted f1 score
        
        # Log
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['val_f1'].append(val_f1_weighted)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1_weighted:.4f}")
        
        # Save best model
        if val_f1_weighted > best_val_f1:
            best_val_f1 = val_f1_weighted
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1_weighted,
                'val_acc': val_acc,
                'config': config
            }, f'best_model_fold{fold_idx}_{representation}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # load best model for testing
    checkpoint = torch.load(f'best_model_fold{fold_idx}_{representation}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1_weighted, test_f1_macro, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, return_details=True
    )
    
    print(f"Test Results (Fold {fold_idx}, {representation})")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 (weighted): {test_f1_weighted:.4f}")
    print(f"F1 (macro): {test_f1_macro:.4f}")
    
    # Per class metrics
    per_class = compute_per_class_metrics(test_labels, test_preds)
    print(f"Per class metrics:")
    for comp_type, metrics in per_class.items():
        # TODO
        print(f"{comp_type}: Acc={metrics['accuracy']:.3f}, "f"F1 = {metrics['f1']:.3f}, Support = {metrics['support']}")
    
    results.update({
        'fold_idx': fold_idx,
        'representation': representation,
        'test_acc': test_acc,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_macro': test_f1_macro,
        'per_class': per_class,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    })
    
    return results


def run_cross_validation(config, representation='star'):
    all_results = []
    
    for fold_idx in range(5):
        results = train_fold(fold_idx, config, representation)
        all_results.append(results)
    
    test_accs = [r['test_acc'] for r in all_results]
    test_f1s = [r['test_f1_weighted'] for r in all_results]
    
    print(f"Cross-Validation Results ({representation})")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Per fold: {[f'{acc:.4f}' for acc in test_accs]}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"fegin_experiments/fegin_results_{representation}_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "cv_results.json", 'w') as f:
        json.dump({
            'config': config,
            'representation': representation,
            'fold_results': all_results,
            'mean_test_acc': np.mean(test_accs),
            'std_test_acc': np.std(test_accs),
            'mean_test_f1': np.mean(test_f1s),
            'std_test_f1': np.std(test_f1s)
        }, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    config = {
        'hidden_channels': 128,
        'num_layers': 3,
        'gnn_type': 'gin',
        'dropout': 0.3,
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'patience': 30,
        'use_descriptors': True
    }
    

    print("FEGIN Component Classification")
    print(f"Config: {config}")

    print("Training on Star-Graph Representation")
    star_results = run_cross_validation(config, representation='star')

    print("Training on Component-Level Representation")
    comp_results = run_cross_validation(config, representation='component')
    
    # Compare
    star_acc = np.mean([r['test_acc'] for r in star_results])
    comp_acc = np.mean([r['test_acc'] for r in comp_results])
    improvement = ((star_acc - comp_acc) / comp_acc) * 100
    

    print("FINAL COMPARISON")

    print(f"Star-graph accuracy: {star_acc:.4f}")
    print(f"Component-level accuracy: {comp_acc:.4f}")
    print(f"Improvement: {improvement:+.2f}%")