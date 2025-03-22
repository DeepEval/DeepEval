import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Primary path
        primary1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        primary2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary1)
        primary3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary2)

        # Branch path
        branch1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)

        # Concatenate both paths
        output_tensor = Concatenate()([primary3, branch2])
        return output_tensor

    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Get the shape of features from Block 1
        shape = input_tensor.shape
        
        # Assume channels_last data format
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 4
        channels_per_group = channels // groups

        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)

        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)

        # Reshape back to original shape
        shuffled = Reshape((height, width, channels))(permuted)
        return shuffled
    
    block2_output = block2(block1_output)
    
    # Final Classification Layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model