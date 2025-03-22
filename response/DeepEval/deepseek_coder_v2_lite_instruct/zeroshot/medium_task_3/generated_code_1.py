import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the inputs
    inputs = Input(shape=input_shape)
    
    # First path (left)
    conv_1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool_1 = MaxPooling2D((2, 2))(conv_1)
    
    # Second path (right)
    conv_2 = Conv2D(64, (3, 3), activation='relu')(pool_1)
    pool_2 = MaxPooling2D((2, 2))(conv_2)
    
    # Third path (right, additional convolution)
    conv_3 = Conv2D(64, (3, 3), activation='relu')(pool_2)
    pool_3 = MaxPooling2D((2, 2))(conv_3)
    
    # Combine the outputs from all paths
    combined = Add()([pool_1, pool_2, pool_3])
    
    # Flatten the combined output
    flatten = Flatten()(combined)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()