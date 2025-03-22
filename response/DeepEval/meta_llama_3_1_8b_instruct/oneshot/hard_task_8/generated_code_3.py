import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Reshape, Permute, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):

        # Primary path
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        # Branch path
        depthwise_conv_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv_branch)

        # Concatenate features along the channel dimension
        output_tensor = Concatenate()([conv1, conv2, conv_branch])

        return output_tensor
    
    block1_output = block(input_layer)
    bath_norm = BatchNormalization()(block1_output)

    # Block 2
    # Reshape the features into four groups
    shape = (28, 28, 3 * 64)
    reshaped_features = Reshape(target_shape=shape)(bath_norm)
    
    # Swap the third and fourth dimensions
    permuted_features = Permute((1, 2, 4, 3))(reshaped_features)
    
    # Reshape the features back to its original shape
    reshaped_features_back = Reshape(target_shape=(28, 28, 3 * 64))(permuted_features)
    
    # Concatenate the features along the channel dimension
    output_tensor = Concatenate()([reshaped_features_back, bath_norm])

    # Flatten the features
    flatten_layer = Flatten()(output_tensor)
    
    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model