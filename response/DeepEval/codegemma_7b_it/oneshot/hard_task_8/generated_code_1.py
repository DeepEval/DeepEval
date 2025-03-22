import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Primary Path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Block 1: Branch Path
    conv4 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Concatenate Block 1 outputs
    block1_output = Concatenate()([conv1, conv2, conv3, conv5])
    batch_norm1 = BatchNormalization()(block1_output)

    # Block 2: Feature Reshaping
    shape_input = keras.backend.int_shape(block1_output)
    reshape_input = Reshape((shape_input[1], shape_input[2], 4, shape_input[3] // 4))(batch_norm1)
    permute_input = Permute((1, 2, 4, 3))(reshape_input)
    reshape_output = Reshape((shape_input[1], shape_input[2], shape_input[3]))(permute_input)

    # Block 2: Fully Connected Layer
    flatten_layer = Flatten()(reshape_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model