import numpy as np

def _find_binning_thresholds(col_data, max_bins):
    col_data = np.asarray(col_data)
    col_data = col_data[~np.isnan(col_data)]
    unique_values = np.unique(col_data)
    
    n_unique_values = len(unique_values)

    if n_unique_values <= max_bins:
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    else:
        percentiles = np.linspace(0, 100, max_bins + 1)
        thresholds = np.percentile(unique_values, percentiles)
        midpoints = (thresholds[:-1] + thresholds[1:]) / 2

    midpoints = midpoints[~np.isinf(midpoints)]
    
    return midpoints

if __name__ == "__main__":
    sample_data = np.array([1, 2, 2, 3, 4, 5, np.nan, 6, 7, 8, 9, 10])
    max_bins = 5
    result = _find_binning_thresholds(sample_data, max_bins)
    print(result)