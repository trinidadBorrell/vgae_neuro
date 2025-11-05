"""Train VGAE/GAE model for adjacency matrix prediction.

Based on:
- DaehanKim (2019) implementation: https://github.com/DaehanKim/vgae_pytorch
- Thomas Kipf Paper: https://arxiv.org/pdf/1611.07308
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.
    
    Args:
        sparse_mx: Scipy sparse matrix
        
    Returns:
        coords: Coordinate list of non-zero entries (N x 2 array)
        values: Values at those coordinates
        shape: Original matrix shape tuple
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    """Normalize adjacency matrix with symmetric normalization.
    
    Applies the normalization: D^(-1/2) * (A + I) * D^(-1/2)
    where D is the degree matrix, A is the adjacency matrix, and I is identity.
    This normalization is standard for Graph Convolutional Networks.
    
    Args:
        adj: Adjacency matrix (scipy sparse format)
        
    Returns:
        Tuple representation of normalized adjacency matrix (coords, values, shape)
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # Add self-loops
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj, val_ratio=0.05, test_ratio=0.10, verbose=False):
    """Split edges into train/validation/test sets for link prediction evaluation.
    
    This function implements the standard edge masking procedure for graph link prediction:
    1. Removes self-loops from the adjacency matrix
    2. Randomly samples edges for validation and test sets (positive examples)
    3. Generates an equal number of non-edges (negative examples) for validation and test
    4. Returns training adjacency with masked edges removed
    
    The masking ensures:
    - No overlap between train/val/test positive edges
    - Negative edges don't exist in the original graph
    - No duplicate edges in any set
    
    Args:
        adj: Original adjacency matrix (scipy sparse format)
        val_ratio: Fraction of edges to use for validation (default: 0.05 = 5%)
        test_ratio: Fraction of edges to use for testing (default: 0.10 = 10%)
        verbose: If True, print debug information during negative edge sampling
        
    Returns:
        adj_train: Training adjacency matrix with test/val edges removed
        train_edges: Array of training edge pairs (N_train x 2)
        val_edges: Array of validation positive edge pairs (N_val x 2)
        val_edges_false: Array of validation negative edge pairs (N_val x 2)
        test_edges: Array of test positive edge pairs (N_test x 2)
        test_edges_false: Array of test negative edge pairs (N_test x 2)
    """
    # Remove self-loops (diagonal elements)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0, "Diagonal should be zero after removing self-loops"

    # Work with upper triangular matrix to avoid counting edges twice (undirected graph)
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]  # Edges from upper triangular part
    edges_all = sparse_to_tuple(adj)[0]  # All edges (for checking existence)
    
    # Calculate split sizes based on provided ratios
    num_test = int(np.floor(edges.shape[0] * test_ratio))
    num_val = int(np.floor(edges.shape[0] * val_ratio))

    # Randomly split edges into train/val/test sets
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        """Check if edge 'a' exists in edge list 'b' with tolerance.
        
        Args:
            a: Single edge as [node_i, node_j]
            b: Array of edges to check against
            tol: Tolerance for numerical comparison (default: 5 decimal places)
            
        Returns:
            True if edge 'a' (or its reverse) exists in 'b', False otherwise
        """
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    # Generate negative test edges (non-existent edges in the graph)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        
        # Skip self-loops
        if idx_i == idx_j:
            continue
        # Skip if edge exists in original graph
        if ismember([idx_i, idx_j], edges_all):
            continue
        # Skip if already sampled (either direction)
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        
        test_edges_false.append([idx_i, idx_j])

    # Generate negative validation edges (non-existent edges in the graph)
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        
        # Skip self-loops
        if idx_i == idx_j:
            continue
        # Skip if edge exists in training set (either direction)
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        # Skip if edge exists in validation set (either direction)
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        # Skip if edge exists in original graph (either direction)
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        # Skip if already sampled (either direction)
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        
        if verbose:
            print(f'Added negative val edge: ({idx_i}, {idx_j})')
        val_edges_false.append([idx_i, idx_j])
    
    if verbose:
        print(f'Validation negative edges: {val_edges_false}')
        print(f'Test negative edges: {test_edges_false}')
        print(f'Total edges in graph: {len(edges_all)}')
    
    # Sanity checks to ensure proper edge masking
    assert ~ismember(test_edges_false, edges_all), "Test negative edges should not exist in graph"
    assert ~ismember(val_edges_false, edges_all), "Val negative edges should not exist in graph"
    assert ~ismember(val_edges, train_edges), "Val edges should not be in training set"
    assert ~ismember(test_edges, train_edges), "Test edges should not be in training set"
    assert ~ismember(val_edges, test_edges), "Val and test edges should not overlap"

    # Reconstruct training adjacency matrix (symmetric, without test/val edges)
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T  # Make symmetric for undirected graph

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def load_data(adj_path, features_path):
    """Load adjacency matrix and feature matrix from .npy files.
    
    Args:
        adj_path: Path to adjacency matrix .npy file
        features_path: Path to feature matrix .npy file
        
    Returns:
        adj: Adjacency matrix (scipy sparse format)
        features: Feature matrix (numpy array)
    """
    adj = np.load(adj_path, allow_pickle=True)
    # Handle case where adjacency is stored as a numpy object (dict or sparse matrix)
    if isinstance(adj, np.ndarray) and adj.dtype == object:
        adj = adj.item()
    
    features = np.load(features_path, allow_pickle=True)
    
    print(f"Loaded data: adj type={type(adj)}, features type={type(features)}")
    print(f"Shapes: adj={adj.shape}, features={features.shape}")
    
    return adj, features


def prepare_data(adj, features):
    """Prepare graph data for training by splitting edges and normalizing.
    
    Args:
        adj: Original adjacency matrix
        features: Node feature matrix
        
    Returns:
        Dictionary containing all necessary tensors and metadata for training
    """
    # Store original adjacency matrix (without self-loops) for evaluation
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
    # Split edges into train/val/test sets
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    
    # Normalize adjacency for input to GNN
    adj_norm = preprocess_graph(adj_train)
    
    # Convert features to sparse tuple format
    num_nodes = adj_train.shape[0]
    features_sparse = sparse_to_tuple(csr_matrix(features).tocoo())
    num_features = features_sparse[2][1]
    features_nonzero = features_sparse[1].shape[0]
    
    print("\nGraph statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Features per node: {num_features}")
    print(f"  Non-zero features: {features_nonzero}")
    print(f"  Train edges: {len(train_edges)}")
    print(f"  Val edges: {len(val_edges)}")
    print(f"  Test edges: {len(test_edges)}")
    
    # Calculate loss weighting (more weight on positive class due to imbalance)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
    
    # Create labels (adjacency with self-loops added)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    # Convert to PyTorch sparse tensors
    adj_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
    )
    
    adj_label_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj_label[0].T),
        torch.FloatTensor(adj_label[1]),
        torch.Size(adj_label[2])
    )
    
    features_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(features_sparse[0].T),
        torch.FloatTensor(features_sparse[1]),
        torch.Size(features_sparse[2])
    )
    
    # Create weight tensor for class imbalance
    weight_mask = adj_label_tensor.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    
    return {
        'adj_norm': adj_norm_tensor,
        'adj_label': adj_label_tensor,
        'features': features_tensor,
        'weight_tensor': weight_tensor,
        'norm': norm,
        'num_features': num_features,
        'num_nodes': num_nodes,
        'adj_orig': adj_orig,
        'val_edges': val_edges,
        'val_edges_false': val_edges_false,
        'test_edges': test_edges,
        'test_edges_false': test_edges_false
    }


# ============================================================================
# Evaluation and Training Functions
# ============================================================================


def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):
    """Calculate ROC-AUC and Average Precision scores for link prediction.
    
    This function evaluates the model's ability to predict edges by:
    1. Computing sigmoid probabilities for both positive and negative edge samples
    2. Combining predictions and ground truth labels
    3. Computing ROC-AUC (measures ranking quality) and AP (area under precision-recall curve)
    
    ROC-AUC measures how well the model ranks positive edges above negative edges.
    Average Precision summarizes the precision-recall curve, emphasizing performance
    on the positive class.
    
    Args:
        edges_pos: Positive edge pairs to evaluate (N_pos x 2 array)
        edges_neg: Negative edge pairs to evaluate (N_neg x 2 array)
        adj_rec: Reconstructed adjacency matrix from model (logits, not probabilities)
        adj_orig: Original ground truth adjacency matrix (for reference)
        
    Returns:
        roc_score: ROC-AUC score (0-1, higher is better)
        ap_score: Average Precision score (0-1, higher is better)
    """
    def sigmoid(x):
        """Sigmoid activation function to convert logits to probabilities."""
        return 1 / (1 + np.exp(-x))

    # Compute predictions for positive edges (edges that exist)
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    # Compute predictions for negative edges (edges that don't exist)
    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))

    # Combine predictions and create binary labels (1 for positive edges, 0 for negative)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    
    # Compute evaluation metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    """Calculate binary accuracy for adjacency matrix reconstruction.
    
    Compares the reconstructed adjacency matrix (thresholded at 0.5) against
    the ground truth adjacency labels. This metric treats adjacency prediction
    as a binary classification problem for each entry in the matrix.
    
    Args:
        adj_rec: Reconstructed adjacency matrix (logits or probabilities from model)
        adj_label: Ground truth adjacency matrix labels (binary)
        
    Returns:
        accuracy: Fraction of correctly predicted entries (0-1)
    """
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def train_epoch(vgae_model, optimizer, data, model_type='GAE'):
    """Train the model for one epoch.
    
    Args:
        vgae_model: The VAE/GAE model
        optimizer: PyTorch optimizer
        data: Dictionary containing training data tensors
        model_type: 'VGAE' or 'GAE' (determines if KL divergence is used)
        
    Returns:
        loss: Training loss value
        train_acc: Training accuracy
        val_roc: Validation ROC-AUC score
        val_ap: Validation Average Precision score
    """
    # Forward pass
    A_pred = vgae_model(data['features'])
    
    # Compute loss
    optimizer.zero_grad()
    reconstruction_loss = data['norm'] * F.binary_cross_entropy(
        A_pred.view(-1),
        data['adj_label'].to_dense().view(-1),
        weight=data['weight_tensor']
    )
    loss = reconstruction_loss
    
    # Add KL divergence for VGAE
    if model_type == 'VGAE':
        kl_divergence = 0.5 / A_pred.size(0) * (
            1 + 2 * vgae_model.logstd - vgae_model.mean**2 - torch.exp(vgae_model.logstd)**2
        ).sum(1).mean()
        loss -= kl_divergence
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    train_acc = get_acc(A_pred, data['adj_label'])
    val_roc, val_ap = get_scores(
        data['val_edges'],
        data['val_edges_false'],
        A_pred,
        data['adj_orig']
    )
    
    return loss.item(), train_acc.item(), val_roc, val_ap


def train_model(vgae_model, optimizer, data, args):
    """Train the VGAE/GAE model for multiple epochs.
    
    Args:
        vgae_model: The VGAE/GAE model
        optimizer: PyTorch optimizer
        data: Dictionary containing all training data
        args: Parsed command-line arguments
        
    Returns:
        Final model after training
    """
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    for epoch in range(args.num_epoch):
        t = time.time()
        
        loss, train_acc, val_roc, val_ap = train_epoch(
            vgae_model, optimizer, data, model_type=args.model
        )
        
        print(f"Epoch: {epoch + 1:04d} | "
              f"Loss: {loss:.5f} | "
              f"Train Acc: {train_acc:.5f} | "
              f"Val ROC: {val_roc:.5f} | "
              f"Val AP: {val_ap:.5f} | "
              f"Time: {time.time() - t:.5f}s")
    
    return vgae_model


def evaluate_model(vgae_model, data):
    """Evaluate the trained model on test set.
    
    Args:
        vgae_model: Trained VGAE/GAE model
        data: Dictionary containing test data
        
    Returns:
        test_roc: Test ROC-AUC score
        test_ap: Test Average Precision score
    """
    with torch.no_grad():
        A_pred = vgae_model(data['features'])
    
    test_roc, test_ap = get_scores(
        data['test_edges'],
        data['test_edges_false'],
        A_pred,
        data['adj_orig']
    )
    
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)
    print(f"Test ROC-AUC: {test_roc:.5f}")
    print(f"Test AP: {test_ap:.5f}")
    print("="*70 + "\n")
    
    return test_roc, test_ap


def save_model(vgae_model, save_dir, model_name='vgae_model.pth'):
    """Save trained model weights.
    
    Args:
        vgae_model: Trained VGAE/GAE model
        save_dir: Directory to save the model
        model_name: Name of the model file
        
    Returns:
        Path to saved model file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / model_name
    torch.save(vgae_model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    return str(model_path)


def visualize_adjacency_matrices(adj_true, adj_pred, save_dir, filename='adjacency_comparison.png'):
    """Create visualization comparing true and predicted adjacency matrices.
    
    Args:
        adj_true: Ground truth adjacency matrix (sparse or dense)
        adj_pred: Predicted adjacency matrix (torch tensor)
        save_dir: Directory to save the plot
        filename: Name of the plot file
        
    Returns:
        Path to saved plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dense numpy arrays
    if sp.issparse(adj_true):
        adj_true = adj_true.toarray()
    
    if torch.is_tensor(adj_pred):
        adj_pred = adj_pred.detach().cpu().numpy()
    
    # Apply sigmoid to predictions if needed (convert logits to probabilities)
    adj_pred_prob = 1 / (1 + np.exp(-adj_pred))
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot true adjacency
    im1 = axes[0].imshow(adj_true, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('True Adjacency Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Node Index')
    axes[0].set_ylabel('Node Index')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot predicted adjacency (probabilities)
    im2 = axes[1].imshow(adj_pred_prob, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Predicted Adjacency Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Node Index')
    axes[1].set_ylabel('Node Index')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot difference (error)
    diff = np.abs(adj_true - adj_pred_prob)
    im3 = axes[2].imshow(diff, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Node Index')
    axes[2].set_ylabel('Node Index')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    plot_path = save_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    return str(plot_path)


def save_results_to_json(results_dict, save_dir, filename='training_results.json'):
    """Save training and evaluation results to JSON file.
    
    Args:
        results_dict: Dictionary containing all results
        save_dir: Directory to save the JSON file
        filename: Name of the JSON file
        
    Returns:
        Path to saved JSON file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = save_dir / filename
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results saved to: {json_path}")
    return str(json_path)


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VGAE/GAE for adjacency matrix prediction',
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
    
    # Training arguments
    parser.add_argument('--num-epoch', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for optimizer')
    
    # Edge masking arguments
    parser.add_argument('--val-ratio', type=float, default=0.05,
                        help='Fraction of edges for validation')
    parser.add_argument('--test-ratio', type=float, default=0.10,
                        help='Fraction of edges for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base directory for saving results')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model weights')
    parser.add_argument('--save-viz', action='store_true', default=True,
                        help='Save visualization of adjacency matrices')
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "="*70)
    print("VGAE/GAE Training for Adjacency Matrix Prediction")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Hidden dimensions: {args.hidden1_dim}, {args.hidden2_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epoch}")
    print(f"Random seed: {args.seed}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    adj, features = load_data(args.adj_path, args.features_path)
    
    # Prepare data for training
    print("\nPreparing data...")
    data = prepare_data(adj, features)
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    import model as model_module
    vgae_model = getattr(model_module, args.model)(data['adj_norm'])
    
    # Initialize optimizer
    from torch.optim import Adam
    optimizer = Adam(vgae_model.parameters(), lr=args.learning_rate)
    
    # Train model
    train_start_time = time.time()
    vgae_model = train_model(vgae_model, optimizer, data, args)
    train_duration = time.time() - train_start_time
    
    # Evaluate on test set
    test_roc, test_ap = evaluate_model(vgae_model, data)
    
    # Get final predictions for visualization and saving
    with torch.no_grad():
        A_pred = vgae_model(data['features'])
    
    # Create output directories
    base_output_dir = Path(args.output_dir)
    model_dir = base_output_dir / 'model'
    viz_dir = base_output_dir / 'visualizations'
    results_dir = base_output_dir / 'metrics'
    
    # Save model weights
    if args.save_model:
        model_filename = f"{args.model}_epoch{args.num_epoch}_lr{args.learning_rate}.pth"
        model_path = save_model(vgae_model, model_dir, model_filename)
    else:
        model_path = None
    
    # Create visualizations
    if args.save_viz:
        viz_filename = f"adjacency_comparison_{args.model}_epoch{args.num_epoch}.png"
        viz_path = visualize_adjacency_matrices(
            data['adj_orig'], 
            A_pred, 
            viz_dir, 
            viz_filename
        )
    else:
        viz_path = None
    
    # Compile results dictionary
    results = {
        'model': {
            'type': args.model,
            'hidden1_dim': args.hidden1_dim,
            'hidden2_dim': args.hidden2_dim,
            'num_nodes': int(data['num_nodes']),
            'num_features': int(data['num_features'])
        },
        'training': {
            'num_epochs': args.num_epoch,
            'learning_rate': args.learning_rate,
            'duration_seconds': float(train_duration),
            'seed': args.seed
        },
        'data': {
            'num_train_edges': len(data['val_edges']) + len(data['test_edges']),
            'num_val_edges': len(data['val_edges']),
            'num_test_edges': len(data['test_edges']),
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio
        },
        'test_metrics': {
            'roc_auc': float(test_roc),
            'average_precision': float(test_ap)
        },
        'files': {
            'model_weights': model_path if model_path else 'not_saved',
            'visualization': viz_path if viz_path else 'not_saved'
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results to JSON
    results_filename = f"results_{args.model}_epoch{args.num_epoch}.json"
    json_path = save_results_to_json(results, results_dir, results_filename)
    
    print("\n" + "="*70)
    print("All results saved successfully!")
    print("="*70)
    if model_path:
        print(f"✓ Model: {model_path}")
    if viz_path:
        print(f"✓ Visualization: {viz_path}")
    print(f"✓ Results JSON: {json_path}")
    print("="*70 + "\n")
    
    print("Training complete!")


if __name__ == '__main__':
    main()