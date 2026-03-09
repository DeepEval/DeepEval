import numpy as np
import pandas as pd

def _find_binning_thresholds(col_data, max_bins):
    # Remove missing values
    col_data = col_data[~pd.isnull(col_data)]
    
    # Sort the data
    col_data = np.sort(col_data)
    
    # Identify distinct values
    distinct_values = np.unique(col_data)
    
    # If the number of distinct values is less than or equal to max_bins, calculate midpoints
    if len(distinct_values) <= max_bins:
        binning_thresholds = (distinct_values[:-1] + distinct_values[1:]) / 2
    else:
        # Calculate approximate midpoint percentiles
        quantiles = np.linspace(0, 100, max_bins + 1)
        binning_thresholds = np.percentile(col_data, quantiles)
    
    # Ensure there are no +inf thresholds
    binning_thresholds = binning_thresholds[np.isfinite(binning_thresholds)]
    
    return binning_thresholds

if __name__ == "__main__":
    # Sample input values
    col_data = np.array([1.2, 2.3, np.nan, 4.5, 5.6, 2.3, 1.2, 3.4, 4.5, 5.6])
    max_bins = 3
    
    # Call the function
    result = _find_binning_thresholds(col_data, max_bins)
    
    # Print the results
    print(result)