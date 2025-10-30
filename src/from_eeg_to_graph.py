from numpy.lib.stride_tricks import sliding_window_view
import mne
import numpy as np
import argparse
from typing import Dict, List, Tuple
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os


class EEGtoGraph:

    # Read results single subject
    @staticmethod
    def load_epochs(main_path: str, subject_id: str, session_num: str, task: str = 'lg'):
        path = f'{main_path}/sub-{subject_id}/ses-{session_num}/eeg/sub-{subject_id}_ses-{session_num}_task-{task}_acq-01_epo.fif'
        epochs = mne.read_epochs(path, verbose=False)
        return epochs.get_data()  # (n_epochs, n_channels, n_times)

    @staticmethod
    def spherical_distance(phi1: float, theta1: float, phi2: float, theta2: float) -> float:
        # Convert to Cartesian coordinates (assuming unit sphere)
        x1 = np.sin(theta1) * np.cos(phi1)
        y1 = np.sin(theta1) * np.sin(phi1)
        z1 = np.cos(theta1)
        
        x2 = np.sin(theta2) * np.cos(phi2)
        y2 = np.sin(theta2) * np.sin(phi2)
        z2 = np.cos(theta2)
        
        # Calculate dot product
        dot_product = x1*x2 + y1*y2 + z1*z2
        
        # Clamp to avoid numerical issues with arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Angular distance
        distance = np.arccos(dot_product)
        
        return distance

    @staticmethod
    def find_k_nearest_sensors(
        coordinates: np.ndarray,
        sensor_names: np.ndarray,
        k: int = 6
    ) -> Dict[str, List[Tuple[str, float]]]:

        n_sensors = coordinates.shape[0]
        
        # Validate inputs
        if coordinates.shape[1] != 2:
            raise ValueError(f"coordinates must have shape (n_sensors, 2), got {coordinates.shape}")
        
        if len(sensor_names) != n_sensors:
            raise ValueError(f"Number of sensor names ({len(sensor_names)}) must match number of coordinates ({n_sensors})")
        
        if k >= n_sensors:
            raise ValueError(f"k ({k}) must be less than the number of sensors ({n_sensors})")
        
        # Calculate pairwise distances
        distance_matrix = np.zeros((n_sensors, n_sensors))
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                phi1, theta1 = coordinates[i]
                phi2, theta2 = coordinates[j]
                
                dist = EEGtoGraph.spherical_distance(phi1, theta1, phi2, theta2)
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
        
        return k_nearest

    # Create adjacency matrix
    @staticmethod
    def adjacency_matrix(
        k: int,
        output_dir: str,
        montage_coordinates_dir: str = '/Users/trinidad.borrell/Documents/Work/PhD/Proyects/VGAE/vgae_neuro/data/biosemi64.txt',
        save: bool = True
    ):
        """
        Create a sparse adjacency matrix based on k-nearest neighbors in spherical coordinates.
        
        Args:
            k: Number of nearest neighbors
            montage_coordinates_dir: Path to the montage coordinates file
            
        Returns:
            Sparse adjacency matrix (scipy.sparse.csr_matrix)
            labels: Array of sensor names
        """
        # Load coordinates and labels
        coords = np.loadtxt(montage_coordinates_dir, usecols=(1, 2))
        labels = np.loadtxt(montage_coordinates_dir, usecols=(0,), dtype=str)
        
        n_sensors = len(labels)
        
        # Find k nearest neighbors
        k_nearest = EEGtoGraph.find_k_nearest_sensors(coords, labels, k)
        
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
            np.save(f'{output_dir}/data/adjency_matrix.npy', adjacency)
        
        return adjacency, labels

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
    def feature_matrix(data_win: np.ndarray, output_dir: str, corr_type: str = 'pearson', save: bool = True
):
        """
        Create feature matrix from windowed data.
        
        Args:
            data_win: Windowed data (n_channels, n_timepoints)
            corr_type: Type of correlation ('pearson')
            
        Returns:
            Feature matrix (n_channels, n_channels)
        """
        if corr_type == 'pearson':
            F = np.corrcoef(data_win)  # Shape: (Nodes, Nodes)

        if save:
            np.save(f'{output_dir}/data/feature_matrix.npy', F)
        return F

    # Plot and save matrices
    @staticmethod
    def plot_matrix(
        matrix,
        title: str,
        output_dir: str,
        filename: str,
        cmap: str = 'viridis',
        figsize: tuple = (10, 8)
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
        main_path: str,
        subject_id: str,
        session_num: str,
        task: str = 'lg',
        window_points: int = 64,
        epoch: int = 0,
        k: int = 6,
        montage_coordinates_dir: str = '/Users/trinidad.borrell/Documents/Work/PhD/Proyects/VGAE/vgae_neuro/data/biosemi64.txt',
        output_dir: str = './output',
        corr_type: str = 'pearson',
        save: bool = True
    ):
        """
        Create adjacency and feature matrices and plot them.
        
        Args:
            main_path: Path to the main data directory
            subject_id: Subject ID
            session_num: Session number
            task: Task name
            window_points: Number of time points in the window
            epoch: Epoch number to process
            k: Number of nearest neighbors for adjacency matrix
            montage_coordinates_dir: Path to montage coordinates file
            output_dir: Directory to save output plots
            corr_type: Type of correlation for feature matrix
            
        Returns:
            adjacency_matrix, feature_matrix, labels
        """
        if save:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f'{output_dir}/data', exist_ok=True)
            os.makedirs(f'{output_dir}/images', exist_ok=True)

        print(f"Processing subject {subject_id}, session {session_num}, epoch {epoch}")
        print(f"Creating graph with k={k} nearest neighbors")
        
        # Create adjacency matrix
        print("\nCreating adjacency matrix...")
        adjacency, labels = EEGtoGraph.adjacency_matrix(k, output_dir, montage_coordinates_dir, save)
        print(f"Adjacency matrix shape: {adjacency.shape}")
        print(f"Number of edges: {adjacency.nnz // 2}")  # Divide by 2 because it's symmetric
        print(f"Sparsity: {1 - adjacency.nnz / (adjacency.shape[0] ** 2):.4f}")
        
        # Create sliding window
        print("\nCreating sliding window...")
        data_win = EEGtoGraph.create_sliding_window(
            main_path, subject_id, session_num, task, window_points, epoch
        )
        
        # Create feature matrix
        print("\nCreating feature matrix...")
        feature_mat = EEGtoGraph.feature_matrix(data_win, output_dir, corr_type, save)
        print(f"Feature matrix shape: {feature_mat.shape}")
        
        # Plot adjacency matrix
        print("\nPlotting adjacency matrix...")
        EEGtoGraph.plot_matrix(
            adjacency,
            f'Adjacency Matrix (k={k})',
            output_dir,
            f'adjacency_matrix_sub-{subject_id}_ses-{session_num}_epoch-{epoch}_k-{k}.png',
            cmap='binary'
        )
        
        # Plot feature matrix
        print("\nPlotting feature matrix...")
        EEGtoGraph.plot_matrix(
            feature_mat,
            f'Feature Matrix ({corr_type.capitalize()} Correlation)',
            output_dir,
            f'feature_matrix_sub-{subject_id}_ses-{session_num}_epoch-{epoch}.png',
            cmap='RdBu_r'
        )
        
        print("\nGraph creation completed successfully!")
        
        return adjacency, feature_mat, labels


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
        '--montage_coordinates_dir',
        type=str,
        default='/Users/trinidad.borrell/Documents/Work/PhD/Proyects/VGAE/vgae_neuro/data/biosemi64.txt',
        help='Path to montage coordinates file'
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
    )
    
    args = parser.parse_args()
    
    # Run the graph creation
    try:
        adjacency, features, labels = EEGtoGraph.create_graph(
            main_path=args.main_path,
            subject_id=args.subject_id,
            session_num=args.session_num,
            task=args.task,
            window_points=args.window_points,
            epoch=args.epoch,
            k=args.k,
            montage_coordinates_dir=args.montage_coordinates_dir,
            output_dir=args.output_dir,
            corr_type=args.corr_type,
            save = args.save
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