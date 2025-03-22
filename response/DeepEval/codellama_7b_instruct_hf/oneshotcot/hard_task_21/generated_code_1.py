import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel axis
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Main path: Depthwise separable convolutional layers with varying kernel sizes
    main_path_output = []
    for kernel_size in [1, 3, 5]:
        conv_layer = Conv2D(64, kernel_size=kernel_size, strides=1, padding='same')(split_layer)
        depthwise_layer = DepthwiseConv2D(64, kernel_size=kernel_size, strides=1, padding='same')(conv_layer)
        main_path_output.append(depthwise_layer)

    # Concatenate outputs from main path
    main_path_output = tf.concat(main_path_output, axis=3)

    # Branch path: 1x1 convolutional layer to align output channels
    branch_path_output = Conv2D(64, kernel_size=1, strides=1, padding='same')(split_layer)

    # Add outputs from main and branch paths
    output_layer = tf.add(main_path_output, branch_path_output)

    # Flatten output and add two fully connected layers
    flatten_layer = Flatten()(output_layer)
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model