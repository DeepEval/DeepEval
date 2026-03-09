import numpy as np

def find_binning_thresholds(col_data, max_bins):
    # Remove missing values
    col_data = col_data[np.isfinite(col_data)]
    
    # Sort the data
    col_data = np.sort(col_data)
    
    # Identify distinct values
    unique_values = np.unique(col_data)
    
    # Calculate midpoints for each bin
    if len(unique_values) <= max_bins:
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    else:
        # Compute approximate midpoint percentiles using the percentile function
        midpoints = np.percentile(col_data, np.linspace(0, 100, max_bins + 1))
    
    # Ensure that there are no +inf thresholds
    midpoints = np.clip(midpoints, -np.inf, np.inf)
    
    # Return the calculated midpoints as the output
    return midpoints

if __name__ == "__main__":
    # Create sample input values
    col_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    max_bins = 3
    # Call the function and print the results
    print(find_binning_thresholds(col_data, max_bins))