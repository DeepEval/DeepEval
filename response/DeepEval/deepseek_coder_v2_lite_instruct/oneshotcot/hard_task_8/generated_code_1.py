import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, DepthwiseConv2D, Reshape, Permute, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=1, padding='same', activation='relu')(conv1x1)
    conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(depthwise_conv3x3)

    branch_depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=1, padding='same', activation='relu')(conv1x1)
    branch_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch_depthwise_conv3x3)

    # Concatenate along the channel dimension
    block1_output = Concatenate(axis=-1)([conv1x1_2, branch_conv1x1])

    # Block 2
    block1_output_shape = block1_output.shape
    reshaped_output = Reshape((block1_output_shape[1], block1_output_shape[2], 2, int(block1_output_shape[3]/2)))(block1_output)
    permuted_output = Permute((1, 2, 4, 3))(reshaped_output)
    final_reshaped_output = Reshape(block1_output_shape[1:3].tolist() + [int(block1_output_shape[3])])(permuted_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(final_reshaped_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model