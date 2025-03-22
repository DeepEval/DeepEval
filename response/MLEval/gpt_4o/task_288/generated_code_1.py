import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Define the number of input features and the number of nodes in the hidden layer
    num_inputs = 10  # You can adjust this based on your data
    num_nodes = 5    # You can adjust this based on your desired model complexity

    # Create a Sequential model
    model = Sequential()
    
    # Add a Dense layer with the specified number of inputs and nodes
    model.add(Dense(units=num_nodes, input_shape=(num_inputs,), activation='relu'))

    # You can compile the model if you want to, depending on your use case
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Output the model summary for validation
    model_summary = model.summary()

    # Return the model summary or any other relevant information
    return model_summary

# Call the method for validation
method()