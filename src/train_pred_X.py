"""Train VGAE/GAE model for feature prediction.

This script supports:
1. A -> X: Predicting features from adjacency matrix
2. X -> A: Predicting adjacency from features (using model.py)

Uses regression loss (MSE) instead of binary classification.
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    """Normalize adjacency matrix with symmetric normalization."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_features(features, val_ratio=0.05, test_ratio=0.10, seed=42):
    """Mask random features for validation and testing.
    
    Args:
        features: Feature matrix (N x D)
        val_ratio: Fraction of features to mask for validation
        test_ratio: Fraction of features to mask for testing
        seed: Random seed
        
    Returns:
        features_train: Training features with validation/test features masked
        val_indices: Validation feature indices
        test_indices: Test feature indices
        val_true: True values for validation features
        test_true: True values for test features
    """
    np.random.seed(seed)
    
    N, D = features.shape
    total_elements = N * D
    
    # Calculate number of elements to mask
    num_test = int(np.floor(total_elements * test_ratio))
    num_val = int(np.floor(total_elements * val_ratio))
    
    # Generate all possible indices
    all_indices = [(i, j) for i in range(N) for j in range(D)]
    np.random.shuffle(all_indices)
    
    # Split into val/test/train
    val_indices = all_indices[:num_val]
    test_indices = all_indices[num_val:num_val + num_test]
    
    # Create masked feature matrix
    features_train = features.copy()
    
    # Store true values and mask them
    val_true = np.array([features[i, j] for i, j in val_indices])
    test_true = np.array([features[i, j] for i, j in test_indices])
    
    for i, j in val_indices:
        features_train[i, j] = 0
    for i, j in test_indices:
        features_train[i, j] = 0
    
    return features_train, val_indices, test_indices, val_true, test_true


def load_data(adj_path, features_path):
    """Load adjacency matrix and feature matrix from .npy files."""
    adj = np.load(adj_path, allow_pickle=True)
    if isinstance(adj, np.ndarray) and adj.dtype == object:
        adj = adj.item()
    
    features = np.load(features_path, allow_pickle=True)
    
    print(f"Loaded data: adj type={type(adj)}, features type={type(features)}")
    print(f"Shapes: adj={adj.shape}, features={features.shape}")
    
    return adj, features


def prepare_data_feat_prediction(adj, features, mask_ratio_val=0.05, mask_ratio_test=0.10, seed=42):
    """Prepare data for feature prediction (A -> X).
    
    Args:
        adj: Adjacency matrix
        features: Feature matrix
        mask_ratio_val: Validation masking ratio
        mask_ratio_test: Test masking ratio
        seed: Random seed
        
    Returns:
        Dictionary containing all necessary tensors for training
    """
    # Remove self-loops from adjacency
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()
    
    # Normalize adjacency for GNN input
    adj_norm = preprocess_graph(adj_orig)
    
    # Mask features for validation/testing
    features_train, val_indices, test_indices, val_true, test_true = mask_features(
        features, val_ratio=mask_ratio_val, test_ratio=mask_ratio_test, seed=seed
    )
    
    num_nodes = adj_orig.shape[0]
    num_features = features.shape[1]
    
    print("\nFeature prediction data statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Features per node: {num_features}")
    print(f"  Masked val features: {len(val_indices)}")
    print(f"  Masked test features: {len(test_indices)}")
    
    # Convert to PyTorch tensors
    adj_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
    )
    
    # For A -> X, we use identity matrix as input (structure only)
    identity_features = np.eye(num_nodes)
    features_tensor = torch.FloatTensor(identity_features)
    features_true = torch.FloatTensor(features)
    
    return {
        'adj_norm': adj_norm_tensor,
        'input_features': features_tensor,  # Identity for A -> X
        'target_features': features_true,    # True features to predict
        'num_features': num_features,
        'num_nodes': num_nodes,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'val_true': torch.FloatTensor(val_true),
        'test_true': torch.FloatTensor(test_true)
    }


def get_regression_metrics(pred, true, indices):
    """Calculate regression metrics (MSE, MAE, R²) for masked features.
    
    Args:
        pred: Predicted feature matrix
        true: True feature matrix
        indices: List of (i, j) indices to evaluate
        
    Returns:
        mse, mae, r2
    """
    pred_values = np.array([pred[i, j] for i, j in indices])
    true_values = np.array([true[i, j] for i, j in indices])
    
    mse = mean_squared_error(true_values, pred_values)
    mae = mean_absolute_error(true_values, pred_values)
    r2 = r2_score(true_values, pred_values)
    
    return mse, mae, r2


def train_epoch_features(model, optimizer, data, model_type='GAE'):
    """Train the model for one epoch (feature prediction).
    
    Args:
        model: The feature prediction model
        optimizer: PyTorch optimizer
        data: Dictionary containing training data
        model_type: 'GAE' or 'VGAE'
        
    Returns:
        loss, train_mse, val_mse, val_mae, val_r2
    """
    # Forward pass
    X_pred = model(data['input_features'])
    
    # Compute MSE loss
    optimizer.zero_grad()
    reconstruction_loss = F.mse_loss(X_pred, data['target_features'])
    loss = reconstruction_loss
    
    # Add KL divergence for VGAE
    if model_type == 'VGAE':
        kl_divergence = 0.5 / X_pred.size(0) * (
            1 + 2 * model.logstd - model.mean**2 - torch.exp(model.logstd)**2
        ).sum(1).mean()
        loss -= kl_divergence
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        X_pred_np = X_pred.detach().cpu().numpy()
        X_true_np = data['target_features'].detach().cpu().numpy()
        
        # Training MSE (full matrix)
        train_mse = mean_squared_error(X_true_np.flatten(), X_pred_np.flatten())
        
        # Validation metrics (masked features)
        val_mse, val_mae, val_r2 = get_regression_metrics(
            X_pred_np, X_true_np, data['val_indices']
        )
    
    return loss.item(), train_mse, val_mse, val_mae, val_r2


def train_model_features(model, optimizer, data, args):
    """Train the feature prediction model for multiple epochs."""
    print("\n" + "="*70)
    print("Starting Feature Prediction Training")
    print("="*70)
    
    for epoch in range(args.num_epoch):
        t = time.time()
        
        loss, train_mse, val_mse, val_mae, val_r2 = train_epoch_features(
            model, optimizer, data, model_type=args.model
        )
        
        print(f"Epoch: {epoch + 1:04d} | "
              f"Loss: {loss:.5f} | "
              f"Train MSE: {train_mse:.5f} | "
              f"Val MSE: {val_mse:.5f} | "
              f"Val MAE: {val_mae:.5f} | "
              f"Val R²: {val_r2:.5f} | "
              f"Time: {time.time() - t:.5f}s")
    
    return model


def evaluate_model_features(model, data):
    """Evaluate the trained feature prediction model on test set."""
    with torch.no_grad():
        X_pred = model(data['input_features'])
    
    X_pred_np = X_pred.detach().cpu().numpy()
    X_true_np = data['target_features'].detach().cpu().numpy()
    
    test_mse, test_mae, test_r2 = get_regression_metrics(
        X_pred_np, X_true_np, data['test_indices']
    )
    
    print("\n" + "="*70)
    print("Test Results (Feature Prediction)")
    print("="*70)
    print(f"Test MSE: {test_mse:.5f}")
    print(f"Test MAE: {test_mae:.5f}")
    print(f"Test R²: {test_r2:.5f}")
    print("="*70 + "\n")
    
    return test_mse, test_mae, test_r2


def save_model(model, save_dir, model_name='model.pth'):
    """Save trained model weights."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / model_name
    torch.save(model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    return str(model_path)


def visualize_features(features_true, features_pred, save_dir, filename='features_comparison.png'):
    """Visualize true vs predicted features."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.is_tensor(features_true):
        features_true = features_true.detach().cpu().numpy()
    if torch.is_tensor(features_pred):
        features_pred = features_pred.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True features
    im1 = axes[0].imshow(features_true, cmap='viridis', aspect='auto')
    axes[0].set_title('True Features', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Node Index')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Predicted features
    im2 = axes[1].imshow(features_pred, cmap='viridis', aspect='auto')
    axes[1].set_title('Predicted Features', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Node Index')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Absolute difference
    diff = np.abs(features_true - features_pred)
    im3 = axes[2].imshow(diff, cmap='Reds', aspect='auto')
    axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Feature Dimension')
    axes[2].set_ylabel('Node Index')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    plot_path = save_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    return str(plot_path)


def save_results_to_json(results_dict, save_dir, filename='results.json'):
    """Save training results to JSON file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = save_dir / filename
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results saved to: {json_path}")
    return str(json_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VGAE/GAE for feature prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--adj-path', type=str,
                        help='Path to adjacency matrix .npy file')
    parser.add_argument('--features-path', type=str,
                        help='Path to feature matrix .npy file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='GAE', choices=['GAE', 'VGAE'],
                        help='Model type: GAE or VGAE')
    parser.add_argument('--hidden1-dim', type=int, default=32,
                        help='Dimension of first hidden layer')
    parser.add_argument('--hidden2-dim', type=int, default=16,
                        help='Dimension of second hidden layer')
    parser.add_argument('--direction', type=str, default='A2X', 
                        choices=['A2X', 'X2A', 'both'],
                        help='Prediction direction: A2X (adj->feat), X2A (feat->adj), or both')
    
    # Training arguments
    parser.add_argument('--num-epoch', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for optimizer')
    
    # Feature masking arguments
    parser.add_argument('--val-ratio', type=float, default=0.05,
                        help='Fraction of features for validation')
    parser.add_argument('--test-ratio', type=float, default=0.10,
                        help='Fraction of features for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results_features',
                        help='Base directory for saving results')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save trained model weights')
    parser.add_argument('--save-viz', action='store_true', default=True,
                        help='Save visualization of features')
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "="*70)
    print("VGAE/GAE Training for Feature Prediction")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Direction: {args.direction}")
    print(f"Hidden dimensions: {args.hidden1_dim}, {args.hidden2_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epoch}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    adj, features = load_data(args.adj_path, args.features_path)
    
    if args.direction in ['A2X', 'both']:
        print("\n" + "="*70)
        print("Training A -> X (Adjacency to Features)")
        print("="*70)
        
        # Prepare data for A -> X
        data = prepare_data_feat_prediction(
            adj, features, 
            mask_ratio_val=args.val_ratio,
            mask_ratio_test=args.test_ratio,
            seed=args.seed
        )
        
        # Initialize model
        print(f"\nInitializing {args.model}_Features model...")
        import model_features as mf
        model_class = getattr(mf, f'{args.model}_Features')
        model = model_class(data['adj_norm'], data['num_features'])
        
        # Initialize optimizer
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        
        # Train model
        train_start = time.time()
        model = train_model_features(model, optimizer, data, args)
        train_duration = time.time() - train_start
        
        # Evaluate
        test_mse, test_mae, test_r2 = evaluate_model_features(model, data)
        
        # Save results
        with torch.no_grad():
            X_pred = model(data['input_features'])
        
        base_output_dir = Path(args.output_dir) / 'A2X'
        
        if args.save_model:
            model_path = save_model(
                model, base_output_dir / 'model',
                f"{args.model}_A2X_epoch{args.num_epoch}.pth"
            )
        
        if args.save_viz:
            viz_path = visualize_features(
                data['target_features'], X_pred,
                base_output_dir / 'visualizations',
                f"features_{args.model}_A2X.png"
            )
        
        results = {
            'direction': 'A2X',
            'model': args.model,
            'test_metrics': {
                'mse': float(test_mse),
                'mae': float(test_mae),
                'r2': float(test_r2)
            },
            'training_time': float(train_duration)
        }
        
        save_results_to_json(
            results, base_output_dir / 'metrics',
            f"results_{args.model}_A2X.json"
        )
    
    if args.direction in ['X2A', 'both']:
        print("\n" + "="*70)
        print("Training X -> A (Features to Adjacency)")
        print("="*70)
        print("This uses the standard model.py (same as train_pred_A.py)")
        print("Please use train_pred_A.py for X -> A prediction")
        print("="*70)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
