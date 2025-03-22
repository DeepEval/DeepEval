import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    # Primary Path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(primary_path)
    
    # Branch Path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch_path)
    
    # Concatenate paths
    block1_output = Concatenate(axis=-1)([primary_path, branch_path])
    
    # Block 2
    # Obtain shape and reshape
    height, width, channels = block1_output.shape[1], block1_output.shape[2], block1_output.shape[3]
    groups = 4
    channels_per_group = channels // groups
    reshaped = Reshape((height, width, groups, channels_per_group))(block1_output)
    
    # Permute dimensions to shuffle channels
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to original dimensions
    shuffled = Reshape((height, width, channels))(permuted)
    
    # Final classification layer
    flatten_layer = Flatten()(shuffled)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model