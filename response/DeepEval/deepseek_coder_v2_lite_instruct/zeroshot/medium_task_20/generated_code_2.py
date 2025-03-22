import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Second path: Two 3x3 convolutions stacked after a 1x1 convolution
    path2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    
    # Third path: Single 3x3 convolution following a 1x1 convolution
    path3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path3 = Conv2D(32, (3, 3), activation='relu', padding='same')(path3)
    
    # Fourth path: Max pooling followed by a 1x1 convolution
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)
    
    # Concatenate the outputs from the four paths
    combined = Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated output
    flattened = Flatten()(combined)
    
    # Dense layer with 128 units
    dense = Dense(128, activation='relu')(flattened)
    
    # Output layer with softmax activation
    outputs = Dense(10, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()