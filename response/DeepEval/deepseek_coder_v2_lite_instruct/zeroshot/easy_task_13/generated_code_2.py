import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # First 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    x = Dropout(0.25)(x)
    
    # Second 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)
    
    # 3x1 convolutional layer
    x = Conv2D(32, (3, 1), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)
    
    # 1x3 convolutional layer
    x = Conv2D(32, (1, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)
    
    # Restore the number of channels to match the input
    x = Conv2D(1, (1, 1), activation='relu')(x)
    
    # Add the processed features with the original input
    x = Add()([x, inputs])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()