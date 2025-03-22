import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, DepthwiseConv2D, Add, Flatten, Dense, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_input[0])
    conv2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_input[1])
    conv3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_input[2])
    block1_output = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Block 2
    reshaped = Reshape((block1_output.shape[1], block1_output.shape[2], 3, block1_output.shape[-1] // 3))(block1_output)
    permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
    block2_output = Reshape((block1_output.shape[1], block1_output.shape[2], block1_output.shape[-1]))(permuted)

    # Block 3
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(block2_output)

    # Main path output
    main_path_output = Add()([depthwise_conv, input_layer])

    # Fully connected output
    flatten_layer = Flatten()(main_path_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model