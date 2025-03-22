# Import necessary packages
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import Accuracy

def method():
    # Define the MLP model
    model = Sequential()
    
    # Input layer and first hidden layer
    model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))
    
    # Second hidden layer
    model.add(Dense(units=32, activation='relu'))
    
    # Output layer
    model.add(Dense(units=output_dim, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=Adam(), 
                  loss=MeanSquaredError(), 
                  metrics=[Accuracy()])
    
    # Return the model
    return model

# Define input and output dimensions for the model
input_dim = 20  # Example input dimension size
output_dim = 3  # Example output dimension size (e.g., for 3 classes)

# Call the method to build the model
mlp_model = method()

# Optionally, print the model summary to validate the architecture
mlp_model.summary()