import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, BatchNormalization, Flatten, Reshape, Permute, Dense

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Primary path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Branch path
    conv4 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Concatenate features from both paths along the channel dimension
    concat_layer = Concatenate(axis=-1)([conv3, conv5])
    
    # Block 2
    reshape_layer = Reshape((-1, 8))(concat_layer)  # Reshape to (height, width, groups, channels_per_group)
    permute_layer = Permute((3, 1, 2, 4))(reshape_layer)  # Swap third and fourth dimensions
    reshape_output = Reshape((28, 28, 4, 1))(permute_layer)  # Reshape back to original shape
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(reshape_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model