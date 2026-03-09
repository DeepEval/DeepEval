import numpy as np
from scipy.stats import gmean

def _find_binning_thresholds(col_data, max_bins):
    # Remove missing values and sort the data
    col_data = col_data[col_data.notnull()]
    col_data = np.sort(col_data)
    
    # Identify distinct values
    distinct_values = np.unique(col_data)
    
    # Calculate the number of distinct values
    n_distinct_values = len(distinct_values)
    
    # If the number of distinct values is less than or equal to max_bins, use midpoints
    if n_distinct_values <= max_bins:
        binning_thresholds = 0.5 * (distinct_values[1:] + distinct_values[:-1])
    else:
        # Calculate approximate midpoint percentiles using the percentile function
        binning_thresholds = np.percentile(col_data, range(max_bins + 1))
    
    # Ensure no +inf thresholds
    binning_thresholds = np.clip(binning_thresholds, 1e-9, None)
    
    return binning_thresholds

if __name__ == "__main__":
    # Sample input values
    col_data = np.random.rand(100)
    max_bins = 10
    
    # Call the function and print the results
    binning_thresholds = _find_binning_thresholds(col_data, max_bins)
    print("Binning Thresholds:", binning_thresholds)