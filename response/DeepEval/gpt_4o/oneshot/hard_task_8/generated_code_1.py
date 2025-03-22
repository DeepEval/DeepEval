import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Feature extraction with multiple paths
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Concatenate paths
    concat_features = Concatenate()([main_path, branch_path])

    # Block 2: Channel shuffling
    # Get the shape of features
    shape = keras.backend.int_shape(concat_features)
    height, width, channels = shape[1], shape[2], shape[3]
    groups = 4  # Define the number of groups
    channels_per_group = channels // groups

    reshaped = Reshape((height, width, groups, channels_per_group))(concat_features)
    permuted = Permute((1, 2, 4, 3))(reshaped)
    shuffled = Reshape((height, width, channels))(permuted)

    # Fully connected layer for classification
    flatten_layer = Flatten()(shuffled)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model