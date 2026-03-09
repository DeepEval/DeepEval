import numpy as np
from scipy.stats import percentileof_rank

def _find_binning_thresholds(col_data, max_bins):
    """
    Creates quantiles from a continuous feature.

    Args:
        col_data: Array-like object representing the continuous feature to bin.
        max_bins: Integer indicating the maximum number of bins to use for non-missing values.

    Returns:
        ndarray of shape (min(max_bins, n_unique_values) - 1) containing increasing numeric values that can be used to separate the bins.
    """
    # Remove missing values
    col_data = col_data[~np.isnan(col_data)]

    # Sort the data
    col_data = np.sort(col_data)

    # Identify distinct values
    unique_values = np.unique(col_data)

    # Calculate midpoints for unique values
    if len(unique_values) <= max_bins:
        midpoints = np.array([(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)])

    # Calculate approximate midpoint percentiles
    else:
        percentiles = percentileof_rank(col_data, np.arange(1, max_bins + 1) / max_bins)
        midpoints = np.interp(np.arange(max_bins), np.arange(max_bins + 1), percentiles)

    # Ensure no +inf thresholds
    midpoints[np.isinf(midpoints)] = np.max(col_data)

    return midpoints

if __name__ == "__main__":
    # Sample input values
    col_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, 10, 11, 12, 13, 14])
    max_bins = 5

    # Call the function and print the results
    thresholds = _find_binning_thresholds(col_data, max_bins)
    print(thresholds)