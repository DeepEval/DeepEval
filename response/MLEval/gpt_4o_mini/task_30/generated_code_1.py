import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def method(input_shape=(32,), num_classes=10):
    # Initialize the Sequential model
    model = Sequential()
    
    # Add input layer and first hidden layer
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    
    # Add second hidden layer
    model.add(Dense(64, activation='relu'))
    
    # Add output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the compiled model
    return model

# Call the generated method for validation
output_model = method()
print(output_model.summary())