import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def method():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2 * X.flatten() + np.random.randn(100) * 2  # Linear relationship with noise
    data = pd.DataFrame({'X': X.flatten(), 'y': y})

    # Initial regression model
    model = sm.OLS(data['y'], sm.add_constant(data['X'])).fit()
    
    # Identify outliers and high leverage points
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    high_leverage_points = influence.hat_matrix_diag > (2 * (model.df_model + 1) / len(data))
    outliers = cooks_d > 4 / len(data)  # Using a common threshold for Cook's distance

    # Create a mask to filter out outliers and high leverage points
    mask = ~(outliers | high_leverage_points)
    
    # Filter the data
    filtered_data = data[mask]
    
    # Run regression again with filtered data
    new_model = sm.OLS(filtered_data['y'], sm.add_constant(filtered_data['X'])).fit()
    
    # Output results
    output = {
        "original_model_summary": model.summary(),
        "filtered_model_summary": new_model.summary(),
        "removed_outliers": data[mask].shape[0],
        "total_data_points": data.shape[0]
    }
    
    return output

# Call the method for validation
output = method()
print(output["original_model_summary"])
print("\n---\n")
print(output["filtered_model_summary"])