"""
Hypervolume (HV) calculation following MOMO_task4.py logic.

This module calculates the hypervolume indicator for multi-objective optimization,
which measures the volume of the objective space dominated by a Pareto front.
"""

import numpy as np
import pygmo as pg


def calculate_hypervolume(fits, reference_point=None):
    """
    Calculate hypervolume for a set of fitness values.
    
    Following MOMO_task4.py logic:
    - Filters out solutions where any objective < 0
    - Negates all objectives (converts maximization to minimization for pygmo)
    - Uses zero vector as reference point by default
    
    Args:
        fits: numpy array of shape (n_solutions, n_objectives)
              Each row is a solution's objective values
        reference_point: numpy array of shape (n_objectives,)
                        Reference point for HV calculation
                        If None, uses zero vector
    
    Returns:
        float: Hypervolume value. Returns 0 if no valid solutions exist.
    
    Example:
        >>> fits = np.array([[0.8, 0.5, 0.6, 0.7],
        ...                  [0.9, 0.4, 0.5, 0.8],
        ...                  [0.7, 0.6, 0.7, 0.6]])
        >>> hv = calculate_hypervolume(fits)
    """
    if len(fits) == 0:
        return 0.0
    
    # Convert to numpy array if not already
    fits = np.array(fits)
    
    # Filter out inf and nan values
    # 过滤掉包含 inf 或 nan 的解
    finite_mask = np.isfinite(fits).all(axis=1)
    fits = fits[finite_mask]
    
    if len(fits) == 0:
        return 0.0
    
    # Filter: only keep solutions where all objectives >= 0
    # This follows the logic: (fit>=[0,0,0,0]).all()
    n_objectives = fits.shape[1]
    valid_mask = (fits >= 0).all(axis=1)
    valid_fits = fits[valid_mask]
    
    if len(valid_fits) == 0:
        return 0.0
    
    # Filter out all-zero solutions (they are on the reference point boundary)
    # 过滤掉全零解（它们正好在参考点边界上，会导致pygmo报错）
    non_zero_mask = ~(valid_fits == 0).all(axis=1)
    valid_fits = valid_fits[non_zero_mask]
    
    if len(valid_fits) == 0:
        return 0.0
    
    # Negate all objectives (pygmo expects minimization problem)
    # This follows: -1.0 * fit for fit in fits
    negated_fits = -1.0 * valid_fits
    
    # Set reference point (default: zero vector)
    if reference_point is None:
        reference_point = np.zeros(n_objectives)
    
    # Calculate hypervolume using pygmo
    try:
        hv_obj = pg.hypervolume(negated_fits)
        dominated_hypervolume = hv_obj.compute(reference_point)
        
        # 检查返回值是否为nan或inf
        if not np.isfinite(dominated_hypervolume):
            print(f"Warning: HV calculation returned {dominated_hypervolume}")
            return 0.0
            
        return dominated_hypervolume
    except Exception as e:
        print(f"Warning: Hypervolume calculation failed: {e}")
        return 0.0


def calculate_hypervolume_per_iteration(fits_list, reference_point=None):
    """
    Calculate hypervolume for each iteration.
    
    Args:
        fits_list: list of numpy arrays, where each array contains
                   fitness values for one iteration
        reference_point: reference point for HV calculation
    
    Returns:
        list: Hypervolume values for each iteration
    
    Example:
        >>> iter1_fits = np.array([[0.8, 0.5], [0.9, 0.4]])
        >>> iter2_fits = np.array([[0.85, 0.55], [0.92, 0.45]])
        >>> hv_values = calculate_hypervolume_per_iteration([iter1_fits, iter2_fits])
    """
    hv_values = []
    for fits in fits_list:
        hv = calculate_hypervolume(fits, reference_point)
        hv_values.append(hv)
    return hv_values


def calculate_hypervolume_from_csv(csv_path, objective_columns, reference_point=None):
    """
    Calculate hypervolume from a CSV file containing optimization results.
    
    Args:
        csv_path: path to CSV file
        objective_columns: list of column names for objectives
        reference_point: reference point for HV calculation
    
    Returns:
        float: Hypervolume value
    
    Example:
        >>> hv = calculate_hypervolume_from_csv(
        ...     'results.csv',
        ...     ['qed', 'gskb', 'sa_nom', 'sim']
        ... )
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    fits = df[objective_columns].values
    
    return calculate_hypervolume(fits, reference_point)


if __name__ == "__main__":
    # Test example following MOMO_task4 (4 objectives: qed, gskb, sa_nom, sim)
    print("Testing HV calculation following MOMO_task4.py logic...")
    
    # Example fitness values (4 objectives)
    test_fits = np.array([
        [0.85, 0.45, 0.60, 0.82],  # Valid solution
        [0.90, 0.35, 0.55, 0.85],  # Valid solution
        [0.80, 0.50, 0.65, 0.78],  # Valid solution
        [-0.1, 0.40, 0.60, 0.80],  # Invalid (negative objective)
    ])
    
    hv = calculate_hypervolume(test_fits)
    print(f"Hypervolume: {hv:.6f}")
    
    # Test with only valid solutions
    valid_fits = test_fits[:3]
    hv_valid = calculate_hypervolume(valid_fits)
    print(f"Hypervolume (valid only): {hv_valid:.6f}")
    
    # Test empty case
    empty_fits = np.array([]).reshape(0, 4)
    hv_empty = calculate_hypervolume(empty_fits)
    print(f"Hypervolume (empty): {hv_empty:.6f}")
