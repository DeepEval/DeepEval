import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Conv1: 1x1 convolution
    conv1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Conv2: 3x3 convolution
    conv2 = Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Conv3: 5x5 convolution
    conv3 = Conv2D(32, (5, 5), activation='relu')(inputs)
    
    # Pooling layer
    pool = MaxPooling2D(pool_size=(3, 3))(inputs)
    
    # Concatenate the features
    concatenated = Concatenate()([conv1, conv2, conv3, pool])
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=inputs, outputs=dense2)
    
    return model

# Example usage
model = dl_model()
model.summary()