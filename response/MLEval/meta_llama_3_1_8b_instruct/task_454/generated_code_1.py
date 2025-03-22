# Import necessary packages
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Define the method to create the shallow network
def method():
    # Define the input and output dimensions
    input_dim = 10
    output_dim = 1
    
    # Create a Sequential model
    model = Sequential()
    
    # Add an input layer
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    
    # Add an output layer
    model.add(Dense(output_dim))
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Return the compiled model
    return model

# Generate a random dataset for validation
np.random.seed(0)
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Create a shallow network
network = method()

# Print the network architecture
print(network.summary())

# Train the network (optional)
# network.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Output the network
output = network
print("Shallow Network:", output)