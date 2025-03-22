import pandas as pd
from sklearn.linear_model import LinearRegression

def method():
    # Data
    data = [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]]

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Number", "Square", "Cube"])

    # Train model
    model = LinearRegression()
    model.fit(df[["Number", "Square"]], df["Cube"])

    # Predict next term
    number = 6
    square = number ** 2
    cube = model.predict([[number, square]])

    # Return output
    return cube

# Call method for validation
output = method()

# Print output
print(output)