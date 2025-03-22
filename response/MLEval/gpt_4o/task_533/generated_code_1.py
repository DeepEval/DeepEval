import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from statsmodels.stats.outliers_influence import OLSInfluence

def method():
    # Generate synthetic data
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10.0)
    X = sm.add_constant(X)  # Add a constant (bias) term
    
    # Initial regression
    model = sm.OLS(y, X).fit()
    influence = OLSInfluence(model)
    
    # Identify outliers and high leverage points
    standardized_residuals = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    outlier_threshold = 2
    leverage_threshold = 2 * (X.shape[1] / X.shape[0])
    
    outliers = np.abs(standardized_residuals) > outlier_threshold
    high_leverage_points = leverage > leverage_threshold
    
    # Combine the masks
    combined_mask = outliers | high_leverage_points
    
    # Remove outliers and high leverage points
    X_clean = X[~combined_mask]
    y_clean = y[~combined_mask]
    
    # Regression after cleaning
    model_clean = sm.OLS(y_clean, X_clean).fit()
    
    # Display results
    print("Initial Model Summary:\n", model.summary())
    print("\nCleaned Model Summary:\n", model_clean.summary())
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Original data and model
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 1], y, label='Original data', color='gray')
    plt.plot(X[:, 1], model.predict(X), color='blue', label='Original model')
    plt.title("Before Removing Outliers")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    # Cleaned data and model
    plt.subplot(1, 2, 2)
    plt.scatter(X_clean[:, 1], y_clean, label='Cleaned data', color='gray')
    plt.plot(X_clean[:, 1], model_clean.predict(X_clean), color='red', label='Cleaned model')
    plt.title("After Removing Outliers")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model_clean

# Call the method for validation
cleaned_model = method()