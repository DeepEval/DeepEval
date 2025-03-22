import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     

    input_layer = keras.Input(shape=(32, 32, 3))

    def depthwise_separable_conv(x, kernel_size):
        return layers.SeparableConv2D(filters=64, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(x)

    main_path = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    main_path_1x1 = depthwise_separable_conv(main_path[0], 1)
    main_path_3x3 = depthwise_separable_conv(main_path[1], 3)
    main_path_5x5 = depthwise_separable_conv(main_path[2], 5)
    main_path_output = layers.Concatenate()([main_path_1x1, main_path_3x3, main_path_5x5])

    branch_path = layers.Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Align the number of output channels of the branch path with that of the main path
    branch_path = layers.Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)

    added_output = layers.Add()([main_path_output, branch_path])

    # Apply global average pooling to reduce the spatial dimensions
    output = layers.GlobalAveragePooling2D()(added_output)
    output = layers.Dense(units=384, activation='relu')(output)
    output = layers.Dense(units=192, activation='relu')(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model