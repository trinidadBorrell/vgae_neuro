import mne
import numpy as np
import argparse
from typing import Dict, List, Tuple
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
import pandas as pd


class EEGtoGraph:

    # Read results single subject
    @staticmethod
    def load_epochs(main_path: str, subject_id: str, session_num: str, task: str = 'lg'):
        path = f'{main_path}/sub-{subject_id}/ses-{session_num}/eeg/sub-{subject_id}_ses-{session_num}_task-{task}_acq-01_epo.fif'
        epochs = mne.read_epochs(path, verbose=False)
        return epochs.get_data()  # (n_epochs, n_channels, n_times)

    @staticmethod
    def cartesian_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points in 2D Cartesian coordinates."""
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance

    @staticmethod
    def find_k_nearest_sensors(
        coords_df: pd.DataFrame,
        k: int = 6
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], np.ndarray]:
        """
        Find k nearest neighbors for each sensor using Cartesian distance.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y']
            k: Number of nearest neighbors
            
        Returns:
            k_nearest: Dictionary mapping sensor names to list of (neighbor_name, distance) tuples
            distance_matrix: Full pairwise distance matrix
        """
        n_sensors = len(coords_df)
        sensor_names = coords_df['label'].values
        X_array = coords_df['x'].values
        Y_array = coords_df['y'].values
        
        # Validate inputs
        if k >= n_sensors:
            raise ValueError(f"k ({k}) must be less than the number of sensors ({n_sensors})")
        
        # Calculate pairwise distances
        distance_matrix = np.zeros((n_sensors, n_sensors))
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                x1, y1 = X_array[i], Y_array[i]
                x2, y2 = X_array[j], Y_array[j]
                
                dist = EEGtoGraph.cartesian_distance(x1, y1, x2, y2)
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric
        
        # Find k nearest neighbors for each sensor
        k_nearest = {}
        
        for i, sensor_name in enumerate(sensor_names):
            # Get distances from this sensor to all others
            distances = distance_matrix[i, :]
            
            # Get indices of k nearest neighbors (excluding itself)
            # argsort gives indices in ascending order
            nearest_indices = np.argsort(distances)[1:k+1]  # Skip index 0 (self)
            
            # Create list of (neighbor_name, distance) tuples
            neighbors = [(sensor_names[idx], distances[idx])
                for idx in nearest_indices]
            
            k_nearest[sensor_name] = neighbors
        
        return k_nearest, distance_matrix

    @staticmethod
    def plot_k_nearest_positions(
        coords_df: pd.DataFrame,
        distance_matrix: np.ndarray,
        k: int = 6,
        output_dir: str = None,
        save: bool = False
    ):
        """
        Plot the k nearest neighbors for each sensor in an 8x8 grid.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y']
            distance_matrix: Pairwise distance matrix
            k: Number of nearest neighbors to highlight
            output_dir: Directory to save plots (if save=True)
            save: Whether to save the plots
        """
        n_sensors = len(coords_df)
        n_rows = int(np.ceil(np.sqrt(n_sensors)))
        n_cols = int(np.ceil(n_sensors / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
        axes = axes.flatten() if n_sensors > 1 else [axes]
        
        for i in range(n_sensors):
            ax = axes[i]
            row = distance_matrix[i, :]
            
            # Get indices of k smallest values (excluding self at index 0)
            smallest_indices = np.argpartition(row, k+1)[:k+1]
            # Remove self if distance is 0
            smallest_indices = smallest_indices[row[smallest_indices] > 0][:k]
            
            # Plot all sensors in black
            ax.plot(coords_df['x'], coords_df['y'], 'o', color='black', markersize=4, alpha=0.3)
            
            # Plot k nearest neighbors in red
            df_neighbours = coords_df.iloc[smallest_indices]
            ax.plot(df_neighbours['x'], df_neighbours['y'], 'o', color='red', markersize=6)
            
            # Plot current sensor in green
            df_sensor = coords_df.iloc[i]
            ax.plot(df_sensor['x'], df_sensor['y'], 'o', color='green', markersize=8)
            
            # Annotate labels
            for idx, label in enumerate(coords_df['label']):
                ax.annotate(label, (coords_df['x'].iloc[idx], coords_df['y'].iloc[idx]), 
                           fontsize=6, alpha=0.7)
            
            ax.set_title(f"{df_sensor['label']}", fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_sensors, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save and output_dir:
            num_electrodes = len(coords_df)
            output_path = os.path.join(output_dir, 'images', f'k_nearest_neighbours_k{k}_{num_electrodes}_electrodes.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved k-nearest neighbors plot to: {output_path}")
            plt.close()
        else:
            plt.show()

    # Create adjacency matrix
    @staticmethod
    def adjacency_matrix(
        coords_df: pd.DataFrame,
        k: int,
        output_dir: str,
        save: bool = True,
        plot_neighbors: bool = False
    ):
        """
        Create a sparse adjacency matrix based on k-nearest neighbors in Cartesian coordinates.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y']
            k: Number of nearest neighbors
            output_dir: Directory to save outputs
            save: Whether to save the adjacency matrix
            plot_neighbors: Whether to plot the k-nearest neighbors visualization
            
        Returns:
            Sparse adjacency matrix (scipy.sparse.csr_matrix)
            labels: Array of sensor names
            distance_matrix: Full pairwise distance matrix
        """
        labels = coords_df['label'].values
        n_sensors = len(labels)
        
        # Check if adjacency matrix already exists
        if save:
            adjacency_npy_path = f'{output_dir}/data/adjacency_matrix_{n_sensors}_electrodes.npy'
            if os.path.exists(adjacency_npy_path):
                print(f"Loading existing adjacency matrix from {adjacency_npy_path}")
                adjacency = np.load(adjacency_npy_path, allow_pickle=True).item()
                # Still need distance matrix for potential neighbor plotting
                _, distance_matrix = EEGtoGraph.find_k_nearest_sensors(coords_df, k)
                return adjacency, labels, distance_matrix
        
        # Find k nearest neighbors
        k_nearest, distance_matrix = EEGtoGraph.find_k_nearest_sensors(coords_df, k)
        
        # Optional: Plot k-nearest neighbors
        if plot_neighbors:
            EEGtoGraph.plot_k_nearest_positions(
                coords_df, distance_matrix, k, output_dir, save
            )
        
        # Create sparse adjacency matrix
        row_indices = []
        col_indices = []
        
        for i, label in enumerate(labels):
            neighbors = k_nearest[label]
            for neighbor_name, _ in neighbors:
                # Find the index of the neighbor
                j = np.where(labels == neighbor_name)[0][0]
                row_indices.append(i)
                col_indices.append(j)
        
        # Create sparse matrix with ones for edges
        data = np.ones(len(row_indices))
        adjacency = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_sensors, n_sensors))
        
        # Make symmetric (if not already)
        adjacency = adjacency + adjacency.T
        adjacency = (adjacency > 0).astype(float)

        if save:
            np.save(f'{output_dir}/data/adjacency_matrix_{n_sensors}_electrodes.npy', adjacency)
        
        return adjacency, labels, distance_matrix

    # Create sliding window
    @staticmethod
    def create_sliding_window(
        main_path: str,
        subject_id: str,
        session_num: str,
        task: str = 'lg',
        window_points: int = 64,
        epoch_num: int = 0
    ):
        data = EEGtoGraph.load_epochs(main_path, subject_id, session_num, task)  # (n_epochs, n_channels, n_times)
        data_win = data[epoch_num, :, :window_points]
        print(f'Data from original shape {data.shape} --> Window: {data_win.shape}')
        return data_win
    
    # Create feature matrix
    @staticmethod
    def feature_matrix(data_win: np.ndarray, output_dir: str, subject_id: str, session_num: str, corr_type: str = 'pearson', save: bool = True):
        """
        Create feature matrix from windowed data.
        
        Args:
            data_win: Windowed data (n_channels, n_timepoints)
            output_dir: Directory to save outputs
            subject_id: Subject ID
            session_num: Session number
            corr_type: Type of correlation ('pearson')
            save: Whether to save the feature matrix
            
        Returns:
            Feature matrix (n_channels, n_channels)
        """
        if corr_type == 'pearson':
            F = np.corrcoef(data_win)  # Shape: (Nodes, Nodes)

        if save:
            np.save(f'{output_dir}/data/feature_matrix_sub{subject_id}_session_{session_num}.npy', F)
        return F

    # Plot and save matrices
    @staticmethod
    def plot_matrix(
        matrix,
        title: str,
        output_dir: str,
        filename: str,
        cmap: str = 'viridis',
        figsize: tuple = (10, 8),
        labels_: list = []
    ):
        """
        Plot a matrix and save it to a file.
        
        Args:
            matrix: Matrix to plot (can be sparse or dense)
            title: Plot title
            output_dir: Directory to save the plot
            filename: Name of the output file
            cmap: Colormap to use
            figsize: Figure size (width, height)
        """        
        # Convert sparse matrix to dense if necessary
        if sp.issparse(matrix):
            matrix_dense = matrix.toarray()
        else:
            matrix_dense = matrix
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix_dense, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Node Index', fontsize=12)
        ax.set_ylabel('Node Index', fontsize=12)
        ax.set_yticks(range(len(labels_)), labels=labels_, rotation=45, fontsize=6)
        ax.set_xticks(range(len(labels_)), labels=labels_, rotation=45, fontsize=6)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', fontsize=12)
        
        # Save figure
        output_path = os.path.join(output_dir, 'images', filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to: {output_path}")

    # Create graph
    @staticmethod
    def create_graph(
        coords_df: pd.DataFrame,
        main_path: str,
        subject_id: str,
        session_num: str,
        task: str = 'lg',
        window_points: int = 154,
        epoch: int = 0,
        k: int = 6,
        output_dir: str = './output',
        corr_type: str = 'pearson',
        save: bool = True,
        plot_neighbors: bool = False
    ):
        """
        Create adjacency and feature matrices and plot them.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y']
            main_path: Path to the main data directory
            subject_id: Subject ID
            session_num: Session number
            task: Task name
            window_points: Number of time points in the window
            epoch: Epoch number to process
            k: Number of nearest neighbors for adjacency matrix
            output_dir: Directory to save output plots
            corr_type: Type of correlation for feature matrix
            save: Whether to save outputs
            plot_neighbors: Whether to plot k-nearest neighbors visualization
            
        Returns:
            adjacency_matrix, feature_matrix, labels, distance_matrix
        """
        if save:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f'{output_dir}/data', exist_ok=True)
            os.makedirs(f'{output_dir}/images', exist_ok=True)

        print(f"Processing subject {subject_id}, session {session_num}, epoch {epoch}")
        print(f"Creating graph with k={k} nearest neighbors")
        
        num_electrodes = len(coords_df)
        
        # Check if adjacency matrix plot already exists
        adjacency_plot_path = os.path.join(output_dir, 'images', f'adjacency_matrix_{num_electrodes}.png')
        adjacency_npy_path = f'{output_dir}/data/adjacency_matrix_{num_electrodes}_electrodes.npy'
        
        if save and os.path.exists(adjacency_plot_path) and os.path.exists(adjacency_npy_path):
            print("\nAdjacency matrix plot and data already exist. Skipping computation.")
            adjacency = np.load(adjacency_npy_path, allow_pickle=True).item()
            labels = coords_df['label'].values
            _, distance_matrix = EEGtoGraph.find_k_nearest_sensors(coords_df, k)
        else:
            # Create adjacency matrix
            print("\nCreating adjacency matrix...")
            adjacency, labels, distance_matrix = EEGtoGraph.adjacency_matrix(
                coords_df, k, output_dir, save, plot_neighbors
            )
            print(f"Adjacency matrix shape: {adjacency.shape}")
            print(f"Number of edges: {adjacency.nnz // 2}")  # Divide by 2 because it's symmetric
            print(f"Sparsity: {1 - adjacency.nnz / (adjacency.shape[0] ** 2):.4f}")
            
            # Plot adjacency matrix
            print("\nPlotting adjacency matrix...")
            EEGtoGraph.plot_matrix(
                adjacency,
                f'Adjacency Matrix (k={k})',
                output_dir,
                f'adjacency_matrix_{num_electrodes}.png',
                cmap='binary',
                labels_=labels
            )
        
        # Create sliding window
        print("\nCreating sliding window...")
        data_win = EEGtoGraph.create_sliding_window(
            main_path, subject_id, session_num, task, window_points, epoch
        )
        
        # Create feature matrix
        print("\nCreating feature matrix...")
        feature_mat = EEGtoGraph.feature_matrix(data_win, output_dir, subject_id, session_num, corr_type, save)
        print(f"Feature matrix shape: {feature_mat.shape}")
        
        # Plot feature matrix
        print("\nPlotting feature matrix...")
        EEGtoGraph.plot_matrix(
            feature_mat,
            f'Feature Matrix ({corr_type.capitalize()} Correlation)',
            output_dir,
            f'feature_matrix_sub-{subject_id}_ses-{session_num}_epoch-{epoch}.png',
            cmap='RdBu_r',
            labels_=labels
        )
        
        print("\nGraph creation completed successfully!")
        
        return adjacency, feature_mat, labels, distance_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Create graph representations from EEG data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--main_path',
        type=str,
        required=True,
        help='Path to the main data directory'
    )
    parser.add_argument(
        '--subject_id',
        type=str,
        required=True,
        help='Subject ID (e.g., "01")'
    )
    parser.add_argument(
        '--session_num',
        type=str,
        required=True,
        help='Session number (e.g., "01")'
    )
    parser.add_argument(
        '--coordinates_file',
        type=str,
        required=True,
        help='Path to biosemi64.txt file with electrode labels'
    )
    
    # Optional arguments
    parser.add_argument(
        '--task',
        type=str,
        default='lg',
        help='Task name'
    )
    parser.add_argument(
        '--window_points',
        type=int,
        default=64,
        help='Number of time points in the sliding window'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help='Epoch number to process'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=6,
        help='Number of nearest neighbors for adjacency matrix'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--corr_type',
        type=str,
        default='pearson',
        choices=['pearson'],
        help='Type of correlation for feature matrix'
    )
    parser.add_argument(
        '--save',
        type=bool,
        default=True,
        help='Whether to save outputs'
    )
    parser.add_argument(
        '--plot_neighbors',
        action='store_true',
        help='Plot k-nearest neighbors visualization (8x8 grid)'
    )
    
    args = parser.parse_args()
    
    # Load coordinates
    try:
        from eeg_positions import get_elec_coords
        
        # Load labels from file
        labels = np.loadtxt(args.coordinates_file, usecols=(0,), dtype=str)
        
        # Get electrode coordinates
        coords_data = get_elec_coords(system='1005', as_mne_montage=False)
        
        # Filter to only include biosemi64 electrodes
        coords_df = coords_data[coords_data['label'].isin(labels)].copy()
        
        print(f"Loaded {len(coords_df)} electrode coordinates")
        
    except Exception as e:
        print(f"\nERROR loading coordinates: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run the graph creation
    try:
        adjacency, features, labels, distance_matrix = EEGtoGraph.create_graph(
            coords_df=coords_df,
            main_path=args.main_path,
            subject_id=args.subject_id,
            session_num=args.session_num,
            task=args.task,
            window_points=args.window_points,
            epoch=args.epoch,
            k=args.k,
            output_dir=args.output_dir,
            corr_type=args.corr_type,
            save=args.save,
            plot_neighbors=args.plot_neighbors
        )
        
        print("\n" + "="*60)
        print("SUCCESS: All operations completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())