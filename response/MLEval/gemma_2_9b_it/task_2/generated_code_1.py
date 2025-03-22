from sklearn.preprocessing import StandardScaler

def method():
  # Sample features (replace with your actual data)
  features = [[1, 2], [3, 4], [5, 6]]

  # Create a StandardScaler object
  scaler = StandardScaler()

  # Fit the scaler to the features
  scaler.fit(features)

  # You can return the fitted scaler object if needed
  return scaler 

# Call the method and store the output
output = method()

# Print the fitted scaler (optional)
print(output)