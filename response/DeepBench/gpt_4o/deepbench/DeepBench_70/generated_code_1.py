import numpy as np

def _find_binning_thresholds(col_data, max_bins):
    col_data = np.array(col_data)
    col_data = col_data[~np.isnan(col_data)]
    unique_values = np.unique(col_data)
    
    if len(unique_values) <= max_bins:
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
    else:
        percentiles = np.linspace(0, 100, min(max_bins, len(unique_values)) + 1)
        midpoints = np.percentile(col_data, percentiles[1:-1])
    
    midpoints = midpoints[np.isfinite(midpoints)]
    return midpoints

if __name__ == "__main__":
    col_data = [1.2, 2.3, 3.7, 2.2, 4.5, 2.2, 3.1, np.nan, 3.9, 4.2, 1.8, 2.9]
    max_bins = 4
    thresholds = _find_binning_thresholds(col_data, max_bins)
    print("Binning thresholds:", thresholds)