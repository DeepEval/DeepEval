import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))
    
    # First branch
    branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    
    # Second branch
    branch2 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    
    # Third branch
    branch3 = Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Combine outputs of all branches using addition
    combined = Add()([branch1, branch2, branch3])
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Fully connected layer
    dense = Dense(128, activation='relu')(flattened)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()