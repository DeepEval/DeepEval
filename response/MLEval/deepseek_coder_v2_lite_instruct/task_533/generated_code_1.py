import numpy as np
import pandas as pd
import statsmodels.api as sm

def method():
    # Generate a sample dataset
    np.random.seed(123)
    n_samples = 100
    x = np.linspace(0, 20, n_samples)
    y = 2 * x + 5 + np.random.normal(0, 4, n_samples)  # Introduce some noise and outliers

    # Create a DataFrame for better visualization and handling
    df = pd.DataFrame({'x': x, 'y': y})

    # Add an outlier
    df.loc[len(df)] = [50, 105]  # This is an outlier in x direction
    df.loc[len(df)] = [6, -10]   # This is an outlier in y direction

    # Fit the initial model
    X = sm.add_constant(df['x'])  # Add a constant term for the intercept
    model = sm.OLS(df['y'], X).fit()
    initial_output = model.summary()

    # Detect and remove outliers
    residuals = model.resid
    df['residuals'] = residuals
    threshold = 2  # Standard deviation threshold for outliers
    df_cleaned = df[(np.abs(residuals) < threshold * residuals.std())]

    # Fit the model again on the cleaned data
    X_cleaned = sm.add_constant(df_cleaned['x'])
    model_cleaned = sm.OLS(df_cleaned['y'], X_cleaned).fit()
    cleaned_output = model_cleaned.summary()

    # Prepare the final output
    output = {
        'initial_model': initial_output,
        'cleaned_model': cleaned_output
    }

    return output

# Call the method for validation
output = method()
print(output['initial_model'])
print(output['cleaned_model'])