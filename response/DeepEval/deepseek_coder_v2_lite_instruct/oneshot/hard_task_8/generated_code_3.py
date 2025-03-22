import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Primary path
    conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
    conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3x3)
    
    # Branch path
    depthwise_conv3x3_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3x3_branch)
    
    # Concatenate features from both paths
    concatenated = Concatenate(axis=-1)([conv1x1_2, conv1x1_branch])
    
    # Block 2
    # Obtain shape and reshape
    reshaped = Reshape((-1, int(concatenated.shape[3]), 1))(concatenated)
    permuted = Permute((1, 2, 4, 3))(reshaped)
    final_reshaped = Reshape((-1, int(permuted.shape[1]), int(permuted.shape[2]), int(permuted.shape[3] // 4)))(permuted)
    
    # Flatten the final output
    flattened = Flatten()(final_reshaped)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model