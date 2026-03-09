import numpy as np

def _find_binning_thresholds(col_data, max_bins):
  col_data = col_data[~np.isnan(col_data)]
  sorted_data = np.sort(col_data)
  n_unique_values = len(np.unique(col_data))
  if n_unique_values <= max_bins:
    return np.array([np.mean([sorted_data[i], sorted_data[i + 1]]) for i in range(0, len(sorted_data) - 1)])
  else:
    return np.percentile(sorted_data, np.linspace(0, 100, max_bins))

if __name__ == "__main__":
  sample_data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, np.nan, 5, 6])
  thresholds = _find_binning_thresholds(sample_data, 5)
  print(thresholds)