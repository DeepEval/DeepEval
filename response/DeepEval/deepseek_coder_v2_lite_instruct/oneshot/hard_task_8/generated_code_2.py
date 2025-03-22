import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Primary path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(depthwise_conv)
    
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    depthwise_branch = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(branch_conv)
    branch_conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(depthwise_branch)
    
    # Concatenate features from both paths
    concatenated = Concatenate(axis=-1)([conv2, branch_conv2])
    
    # Block 2
    # Get shape and reshape for channel shuffling
    reshaped_shape = keras.backend.int_shape(concatenated)
    reshaped = Reshape((reshaped_shape[1], reshaped_shape[2], 2, int(reshaped_shape[3]/2)))(concatenated)
    
    # Swap third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Flatten and feed into fully connected layers
    flattened = Flatten()(permuted)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model