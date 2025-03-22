import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_dnn():
    # Create a sequential model, which is a linear stack of layers
    model = Sequential()
    
    # Add the first layer, which takes the input data and transforms it
    model.add(Dense(32, input_dim=8, activation='relu'))  # Input shape is 8 for example purposes
    
    # Add a second layer, which further processes the data
    model.add(Dense(64, activation='relu'))
    
    # Add a third layer, which might be the output layer if we're doing a classification
    model.add(Dense(1, activation='sigmoid'))  # For binary classification, you might use 'sigmoid' or 'binary_crossentropy'
    
    # Compile the model with an optimizer and loss function suitable for the task
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # For binary classification
    
    return model

# Call the function to build the DNN
def method():
    # Build the DNN
    dnn = build_dnn()
    
    # If you need to return the output, you can do so here. However, since this is a function to build a DNN,
    # the output is the model itself, which can be used for further predictions or training.
    # return dnn
    
    # For validation purposes, you might print the model summary, which gives a high-level summary of the model's structure
    dnn.summary()

# Call the method for validation
method()