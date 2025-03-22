# Import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the function to build the DNN
def build_dnn(input_shape, num_hidden_layers, units_per_layer):
    """
    Builds a simple Deep Neural Network (DNN) with multiple hidden layers.
    
    Parameters:
    input_shape (tuple): The shape of the input data.
    num_hidden_layers (int): The number of hidden layers in the DNN.
    units_per_layer (int or list): The number of units in each hidden layer. 
                                   If it's an integer, the same number of units will be used in all hidden layers.
                                   If it's a list, the number of units in each hidden layer will be specified in the list.
    
    Returns:
    model (object): The built DNN model.
    """
    # Initialize the DNN model using the Sequential API
    model = Sequential()
    
    # Add the input layer
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    
    # Add the hidden layers
    if isinstance(units_per_layer, int):
        for _ in range(num_hidden_layers - 1):
            model.add(Dense(units_per_layer, activation='relu'))
    elif isinstance(units_per_layer, list):
        if len(units_per_layer)!= num_hidden_layers - 1:
            raise ValueError("The number of units per layer does not match the number of hidden layers.")
        for units in units_per_layer:
            model.add(Dense(units, activation='relu'))
    else:
        raise ValueError("The units per layer must be an integer or a list.")
    
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Test the function
input_shape = (784,)
num_hidden_layers = 3
units_per_layer = 512

model = build_dnn(input_shape, num_hidden_layers, units_per_layer)
print(model.summary())