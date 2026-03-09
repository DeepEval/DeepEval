import numpy as np
import pandas as pd
from scipy import stats

def _find_binning_thresholds(col_data, max_bins):
    # Remove missing values
    valid_data = col_data[~np.isnan(col_data)]
    
    # Sort the data
    sorted_data = np.sort(valid_data)
    
    # Identify distinct values
    _, unique_idx, _ = np.unique(sorted_data, return_index=True, return_inverse=True)
    n_unique_values = len(np.unique(unique_idx))
    
    # Calculate midpoints between consecutive distinct values
    if n_unique_values <= max_bins:
        midpoints = np.zeros(n_unique_values - 1)
        for i in range(n_unique_values - 1):
            midpoints[i] = (sorted_data[unique_idx[i] + 1] + sorted_data[unique_idx[i]]) / 2
    else:
        midpoints = stats.percentileofscore(sorted_data, sorted_data,'midpoint')
    
    # Ensure there are no +inf thresholds
    midpoints = np.where(midpoints == np.inf, np.nan, midpoints)
    
    # Return the calculated midpoints
    return midpoints[:min(max_bins, n_unique_values) - 1]

if __name__ == "__main__":
    # Create sample input values
    col_data = np.array([1, 2, np.nan, 3, 4, 5, 6, np.nan, 7, 8, 9])
    
    # Call the function and print the results
    print(_find_binning_thresholds(col_data, 5))