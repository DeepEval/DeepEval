import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, DepthwiseConv2D, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1_1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(main_path_input[0])
    conv1_3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(main_path_input[1])
    conv1_5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(main_path_input[2])
    main_path_output = Concatenate(axis=-1)([conv1_1, conv1_3, conv1_5])

    # Branch path
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path_output = conv2_1

    # Addition of main and branch paths
    added_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flattened_output = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model