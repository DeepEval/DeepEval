import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Feature extraction path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path1 = MaxPooling2D((2, 2))(path1)
    
    # Feature extraction path 2: sequence of convolutions
    path2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path2)
    path2 = MaxPooling2D((2, 2))(path2)
    
    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])
    
    # Main path: 1x1 convolution to align the output dimensions
    main_path = Conv2D(64, (1, 1), activation='relu')(concatenated)
    
    # Add a branch to the input
    branch = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Merge the outputs of the main path and the branch through addition
    merged = tf.keras.layers.add([main_path, branch])
    
    # Flatten the output
    flattened = Flatten()(merged)
    
    # Add fully connected layers for classification
    outputs = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(outputs)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()