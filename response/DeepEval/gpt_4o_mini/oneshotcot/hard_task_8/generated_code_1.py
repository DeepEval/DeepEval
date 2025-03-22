import keras
from keras.layers import Input, Conv2D, Dense, Concatenate, Reshape, Permute, Activation
from keras.layers import DepthwiseConv2D, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(primary_path)
    
    # Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)
    
    # Concatenate both paths
    block1_output = Concatenate(axis=-1)([primary_path, branch_path])
    
    # Block 2
    # Obtain shape and reshape into groups
    block1_shape = keras.backend.int_shape(block1_output)
    # Assuming that the number of filters is divisible by 4 for reshaping
    groups = 4
    channels_per_group = block1_shape[-1] // groups
    reshaped_output = Reshape((block1_shape[1], block1_shape[2], groups, channels_per_group))(block1_output)
    
    # Permute dimensions to shuffle channels
    permuted_output = Permute((0, 1, 3, 2, 4))(reshaped_output)
    
    # Reshape back to original shape
    shuffled_output = Reshape((block1_shape[1], block1_shape[2], block1_shape[-1]))(permuted_output)
    
    # Flatten the output for the dense layers
    flatten_layer = Flatten()(shuffled_output)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model