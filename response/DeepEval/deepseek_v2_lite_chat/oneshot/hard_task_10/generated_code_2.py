import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction path 1: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Feature extraction path 2: 1x1, 1x7, and 7x1 convolutions
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(7, 1), padding='valid')(conv2)
    conv2 = Conv2D(filters=32, kernel_size=(1, 7), padding='valid')(conv2)
    
    # Concatenate the outputs from both paths
    concat = Concatenate(axis=-1)([conv1, conv2])
    
    # Additional 1x1 convolution for dimension alignment
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    
    # Branch to input for direct comparison
    branch = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    
    # Add the outputs of the main path and the branch
    combined = keras.layers.add([conv3, branch])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model