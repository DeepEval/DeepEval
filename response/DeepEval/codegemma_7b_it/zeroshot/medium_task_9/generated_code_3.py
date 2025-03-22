import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def basic_block(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(
        [
            layers.Conv2D(filters, size, strides=1, padding='same',
                            kernel_initializer=initializer,
                            use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, size, strides=1, padding='same',
                            kernel_initializer=initializer,
                            use_bias=False),
            layers.BatchNormalization()
        ]
    )
    return result

def feature_fusion(main_output, branch_output):
    result = layers.Add()([main_output, branch_output])
    result = layers.Activation('relu')(result)
    return result


def dl_model():

    inputs = keras.Input(shape=(32, 32, 3))
    initializer = tf.random_normal_initializer(0., 0.02)

    # Initial convolutional layer
    x = layers.Conv2D(16, 3, strides=1, padding='same',
                        kernel_initializer=initializer,
                        use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Main path
    main_output = basic_block(16, 3)
    main_output = basic_block(16, 3)(main_output)

    # Branch path
    branch_output = layers.Conv2D(16, 3, strides=1, padding='same',
                                kernel_initializer=initializer,
                                use_bias=False)(inputs)
    branch_output = layers.BatchNormalization()(branch_output)
    branch_output = layers.Activation('relu')(branch_output)
    branch_output = layers.Conv2D(16, 3, strides=1, padding='same',
                                kernel_initializer=initializer,
                                use_bias=False)(branch_output)
    branch_output = layers.BatchNormalization()(branch_output)
    branch_output = layers.Activation('relu')(branch_output)

    # Feature fusion
    output = feature_fusion(main_output, branch_output)

    # Pooling and fully connected layer
    output = layers.AveragePooling2D(4)(output)
    output = layers.Flatten()(output)
    outputs = layers.Dense(10, kernel_initializer=initializer)(output)

    model = keras.Model(inputs, outputs)
    return model