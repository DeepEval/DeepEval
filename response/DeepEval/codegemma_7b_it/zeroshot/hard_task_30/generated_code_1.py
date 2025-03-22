from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, concatenate, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    """
    Residual block implementation for the dual-path structure.
    """
    shortcut = x
    if conv_shortcut:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='valid')(shortcut)
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = add([x, shortcut])
    return x

def depthwise_separable_block(x, filters, kernel_size):
    """
    Depthwise separable convolutional block.
    """
    x = DepthwiseConv2D(kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return x

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    main_path = Conv2D(64, (3, 3), strides=1, padding='same')(input_img)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Conv2D(64, (3, 3), strides=1, padding='same')(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)

    branch_path = Conv2D(64, (1, 1), strides=1, padding='valid')(input_img)

    x = add([main_path, branch_path])
    x = Activation('relu')(x)

    # Block 2: Channel split and depthwise separable convolutions
    x = Lambda(lambda y: tf.split(y, 3, axis=3))(x)
    group_1 = depthwise_separable_block(x[0], 64, (1, 1))
    group_2 = depthwise_separable_block(x[1], 64, (3, 3))
    group_3 = depthwise_separable_block(x[2], 64, (5, 5))

    x = concatenate([group_1, group_2, group_3])

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)

    # Model definition
    model = Model(inputs=input_img, outputs=output)

    return model