# Import necessary packages
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np

# Generate random data for training and testing (replace with your actual data)
np.random.seed(0)
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)
X_test = np.random.rand(20, 10)
y_test = np.random.rand(20, 1)

# Define the method function
def method():
    """
    Builds a sequential model with dropout to reduce overfitting.
    
    Returns:
    output: The trained model
    """
    
    # Create a sequential model
    model = Sequential()
    
    # Add input layer with 10 units and ReLU activation
    model.add(Dense(64, activation='relu', input_shape=(10,)))
    
    # Add hidden layer with 32 units, ReLU activation, and dropout 0.2
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Add hidden layer with 16 units, ReLU activation, and dropout 0.2
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    
    # Add output layer with 1 unit and linear activation
    model.add(Dense(1, activation='linear'))
    
    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    
    # Train the model on the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate the model on the test data
    loss = model.evaluate(X_test, y_test, verbose=0)
    
    # Return the trained model and the test loss
    return model, loss

# Call the method function for validation
output = method()

# Print the test loss
print("Test Loss:", output[1])