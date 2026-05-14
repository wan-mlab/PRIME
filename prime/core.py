import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, coo_matrix, issparse
from typing import List, Tuple, Dict, Optional, Union
import gc

def find_multibatch_mnn_graph(
    X_proj: np.ndarray,
    batch_labels: np.ndarray,
    k_neighbors: int = 20,
    chunk_pairs: bool = True,
    max_pairs_per_chunk: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-optimized MNN finding across all batches.
    
    Key optimizations:
    1. No persistent storage of NN models
    2. Direct set operations instead of sparse matrices
    3. Optional chunking of batch pair processing
    """
    n_cells = X_proj.shape[0]
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    
    # Pre-compute batch indices only (not models)
    batch_indices = {batch: np.where(batch_labels == batch)[0] 
                    for batch in unique_batches}
    
    # Convert to float32 if not already
    if X_proj.dtype != np.float32:
        X_proj = X_proj.astype(np.float32, copy=False)
    
    mnn_rows_total = []
    mnn_cols_total = []
    
    # Sort batches for consistent iteration
    unique_batches = np.sort(unique_batches)
    
    # Generate all batch pairs
    batch_pairs = [(unique_batches[i], unique_batches[j]) 
                   for i in range(n_batches) 
                   for j in range(i + 1, n_batches)]
    
    # Process in chunks if requested (for very large batch counts)
    if chunk_pairs and len(batch_pairs) > max_pairs_per_chunk:
        for chunk_start in range(0, len(batch_pairs), max_pairs_per_chunk):
            chunk_end = min(chunk_start + max_pairs_per_chunk, len(batch_pairs))
            chunk_pairs = batch_pairs[chunk_start:chunk_end]
            
            rows, cols = _process_batch_pairs(
                X_proj, batch_indices, chunk_pairs, k_neighbors
            )
            
            if len(rows) > 0:
                mnn_rows_total.extend([rows, cols])  # Include symmetric
                mnn_cols_total.extend([cols, rows])
            
            # Force garbage collection between chunks
            gc.collect()
    else:
        rows, cols = _process_batch_pairs(
            X_proj, batch_indices, batch_pairs, k_neighbors
        )
        
        if len(rows) > 0:
            mnn_rows_total.extend([rows, cols])
            mnn_cols_total.extend([cols, rows])
    
    if not mnn_rows_total:
        return np.array([]), np.array([])
    
    # Concatenate all MNN pairs
    mnn_rows = np.concatenate(mnn_rows_total)
    mnn_cols = np.concatenate(mnn_cols_total)
    
    return mnn_rows, mnn_cols


def _process_batch_pairs(
    X_proj: np.ndarray,
    batch_indices: dict,
    batch_pairs: list,
    k_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a chunk of batch pairs to find MNNs.
    Uses set operations instead of sparse matrices for memory efficiency.
    """
    all_mnn_rows = []
    all_mnn_cols = []
    
    for b1, b2 in batch_pairs:
        idx_b1 = batch_indices[b1]
        idx_b2 = batch_indices[b2]
        
        # Skip if either batch is too small
        if len(idx_b1) < 2 or len(idx_b2) < 2:
            continue
        
        # Find neighbors B1 -> B2 (on-demand, no model storage)
        k_b2 = min(k_neighbors, len(idx_b2))
        nn_b2 = NearestNeighbors(n_neighbors=k_b2, algorithm='brute', 
                                 metric='euclidean', n_jobs=-1)
        nn_b2.fit(X_proj[idx_b2])
        _, match_b1_to_b2 = nn_b2.kneighbors(X_proj[idx_b1])
        
        # Find neighbors B2 -> B1
        k_b1 = min(k_neighbors, len(idx_b1))
        nn_b1 = NearestNeighbors(n_neighbors=k_b1, algorithm='brute',
                                 metric='euclidean', n_jobs=-1)
        nn_b1.fit(X_proj[idx_b1])
        _, match_b2_to_b1 = nn_b1.kneighbors(X_proj[idx_b2])
        
        # Find mutual neighbors using set operations
        mnn_pairs = _find_mutual_neighbors_sets(
            match_b1_to_b2, match_b2_to_b1
        )
        
        if len(mnn_pairs) > 0:
            # Map to global indices
            local_b1_idx, local_b2_idx = mnn_pairs[:, 0], mnn_pairs[:, 1]
            global_rows = idx_b1[local_b1_idx]
            global_cols = idx_b2[local_b2_idx]
            
            all_mnn_rows.append(global_rows)
            all_mnn_cols.append(global_cols)
    
    if all_mnn_rows:
        return np.concatenate(all_mnn_rows), np.concatenate(all_mnn_cols)
    return np.array([]), np.array([])


def _find_mutual_neighbors_sets(
    match_b1_to_b2: np.ndarray,
    match_b2_to_b1: np.ndarray
) -> np.ndarray:
    """
    Find mutual nearest neighbors using set operations.
    More memory efficient than sparse matrix multiplication.
    """
    n_b1 = match_b1_to_b2.shape[0]
    n_b2 = match_b2_to_b1.shape[0]
    
    # Create forward mapping: B1 -> B2
    forward_edges = set()
    for i in range(n_b1):
        for j in match_b1_to_b2[i]:
            forward_edges.add((i, j))
    
    # Check reverse mapping: B2 -> B1
    mutual_pairs = []
    for j in range(n_b2):
        for i in match_b2_to_b1[j]:
            if (i, j) in forward_edges:
                mutual_pairs.append([i, j])
    
    return np.array(mutual_pairs) if mutual_pairs else np.array([]).reshape(0, 2)

def build_consensus_graph(
    X: Optional[Union[np.ndarray, csr_matrix]],
    batch_labels: np.ndarray,
    projections: Optional[List[np.ndarray]] = None,
    n_projections: int = 10,
    target_dim: int = 50,
    k_neighbors: int = 20,
    consensus_threshold: float = 0.4,
    random_state: Optional[int] = None,
    use_incremental: bool = True
) -> csr_matrix:
    """
    Build consensus graph with incremental updates to avoid memory peaks.
    """
    n_cells = batch_labels.shape[0]
    
    if projections is not None:
        print(f"Using {len(projections)} pre-computed projections...")
        n_projections = len(projections)
    else:
        print(f"Generating {n_projections} random projections on the fly...")
        if X is None:
            raise ValueError("X must be provided if projections are not provided.")
        X_norm = normalize(X, axis=1)
    
    if use_incremental:
        # Use dictionary for incremental edge counting
        edge_counts = {}
        
        for i in range(n_projections):
            # Get or generate projection
            if projections is not None:
                X_proj = projections[i]
            else:
                seed = None if random_state is None else random_state + i
                rp = GaussianRandomProjection(n_components=target_dim, random_state=seed)
                X_proj = rp.fit_transform(X_norm)
            
            # Find MNNs using optimized function
            rows, cols = find_multibatch_mnn_graph(
                X_proj, batch_labels, k_neighbors
            )
            
            # Update edge counts
            for r, c in zip(rows, cols):
                edge_counts[(r, c)] = edge_counts.get((r, c), 0) + 1
            
            # Clean up projection if generated
            if projections is None:
                del X_proj
                gc.collect()
        
        # Build final consensus matrix
        valid_edges = [(k, v/n_projections) for k, v in edge_counts.items() 
                       if v/n_projections >= consensus_threshold]
        
        if valid_edges:
            edges, weights = zip(*valid_edges)
            rows, cols = zip(*edges)
            consensus = csr_matrix((weights, (rows, cols)), 
                                  shape=(n_cells, n_cells), dtype=np.float32)
        else:
            consensus = csr_matrix((n_cells, n_cells), dtype=np.float32)
    
    else:
        # Original implementation (kept for compatibility)
        all_rows = []
        all_cols = []
        
        for i in range(n_projections):
            if projections is not None:
                X_proj = projections[i]
            else:
                seed = None if random_state is None else random_state + i
                rp = GaussianRandomProjection(n_components=target_dim, random_state=seed)
                X_proj = rp.fit_transform(X_norm)
            
            rows, cols = find_multibatch_mnn_graph(
                X_proj, batch_labels, k_neighbors
            )
            
            if len(rows) > 0:
                all_rows.append(rows)
                all_cols.append(cols)
        
        if all_rows:
            all_rows = np.concatenate(all_rows)
            all_cols = np.concatenate(all_cols)
            data = np.ones(len(all_rows), dtype=np.float32)
            consensus = coo_matrix((data, (all_rows, all_cols)), 
                                  shape=(n_cells, n_cells)).tocsr()
            consensus.data /= n_projections
            mask = consensus.data >= consensus_threshold
            consensus.data = consensus.data[mask]
            consensus.indices = consensus.indices[mask]
            consensus.eliminate_zeros()
        else:
            consensus = csr_matrix((n_cells, n_cells), dtype=np.float32)
    
    print(f"Consensus graph: {consensus.nnz} edges")
    return consensus


def _compute_smoothing_matrix(
    X: Union[np.ndarray, csr_matrix], 
    sigma: float = 1.0,
    k_smooth: int = 15
) -> csr_matrix:
    """
    Computes the sparse smoothing matrix based on original data structure.
    """
    print("Computing smoothing kernel...")
    X_norm = normalize(X, axis=1)
    nn = NearestNeighbors(n_neighbors=k_smooth, metric='euclidean', n_jobs=-1)
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)
    
    # Gaussian kernel weights: exp(-dist^2 / sigma)
    weights = np.exp(-distances**2 / sigma)

    # if sigma is very small, weights can be extremely small;
    
    # Normalize weights to sum to 1 per row
    # Add epsilon to avoid division by zero
    weight_sums = weights.sum(axis=1)[:, np.newaxis] + 1e-10
    weights /= weight_sums
    
    # Create sparse matrix
    n_cells = X.shape[0]
    row_ind = np.repeat(np.arange(n_cells), k_smooth)
    col_ind = indices.flatten()
    data = weights.flatten()
    
    return csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_cells))


def ensemble_mnn_correct(
    adata: AnnData,
    batch_key: str,
    projection_keys: Optional[List[str]] = None,
    n_projections: int = 10,
    target_dim: int = 50,
    k_neighbors: int = 20,
    consensus_threshold: float = 0.4,
    sigma: float = 0.1,
    random_state: int = 42,
    key_added: Optional[str] = None,
    inplace: bool = True,
    chunk_size: int = 2000  # Size of gene chunks
) -> Optional[AnnData]:
    """
    Main pipeline function with memory-optimized chunk processing.
    """
    if not inplace:
        adata = adata.copy()
        
    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    X = adata.X
    batch_labels = adata.obs[batch_key].values
    n_cells, n_genes = X.shape
    
    # Handle pre-computed projections
    projections = None
    if projection_keys:
        missing_keys = [k for k in projection_keys if k not in adata.obsm]
        if missing_keys:
            raise KeyError(f"Missing projection keys: {missing_keys}")
        projections = [adata.obsm[k] for k in projection_keys]
    
    # 1. Build Consensus Graph
    consensus_graph = build_consensus_graph(
        X, batch_labels, projections, n_projections, target_dim, 
        k_neighbors, consensus_threshold, random_state
    )
    
    # 2. Prepare Correction Data Structures
    rows, cols = consensus_graph.nonzero()
    weights = consensus_graph.data
    
    # Filter Cross-batch only
    b_rows = batch_labels[rows]
    b_cols = batch_labels[cols]
    mask = b_rows != b_cols
    rows = rows[mask]
    cols = cols[mask]
    weights = weights[mask]
    
    if len(rows) == 0:
        print("Warning: No cross-batch MNNs found. Data unchanged.")
        if not inplace: return adata
        return
    
    # Pre-calculate weight sums for normalization
    weight_sums = np.zeros(n_cells)
    np.add.at(weight_sums, rows, weights)
    
    # 3. Build Smoothing Matrix
    smoothing_mat = _compute_smoothing_matrix(X, sigma=sigma)
    
    # 4. Apply Correction in Chunks (Gene-wise)
    # This prevents allocating massive dense matrices (N_cells * N_genes)
    print(f"Applying correction in chunks of {chunk_size} genes...")
    
    # Prepare output matrix
    # We assume the output needs to be dense as batch correction destroys sparsity
    if key_added or not inplace:
        X_corrected = np.zeros((n_cells, n_genes), dtype=X.dtype)
    else:
        # If inplace and no key_added, we overwrite X. 
        # If X is sparse, this will densify it, which might be slow.
        if issparse(adata.X):
            print("Note: transforming sparse X to dense in-place.")
            adata.X = adata.X.toarray()
        X_corrected = adata.X

    # if n_genes < chunk_size, we can process all at once without chunking
    if n_genes < chunk_size:
        chunk_size = n_genes

    for i in range(0, n_genes, chunk_size):
        end = min(i + chunk_size, n_genes)
        
        # Extract chunk (densify if needed for calculation)
        if issparse(X):
            X_chunk = X[:, i:end].toarray()
        else:
            X_chunk = X[:, i:end]
            
        # Calculate Correction for Chunk
        raw_correction_chunk = np.zeros_like(X_chunk, dtype=np.float64)
        
        # Pull vectors: X[cols] - X[rows]
        diffs = X_chunk[cols] - X_chunk[rows]
        weighted_diffs = diffs * weights[:, np.newaxis]
        
        np.add.at(raw_correction_chunk, rows, weighted_diffs)
        
        # Normalize
        has_correction = weight_sums > 0
        raw_correction_chunk[has_correction] /= weight_sums[has_correction][:, np.newaxis]
        
        # Smooth Chunk
        final_correction_chunk = smoothing_mat.dot(raw_correction_chunk)
        
        # Apply
        X_corrected[:, i:end] = X_chunk + final_correction_chunk

    return X_corrected