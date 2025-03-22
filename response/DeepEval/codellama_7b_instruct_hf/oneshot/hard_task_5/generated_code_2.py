import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)

    # Block 1: Channel split
    block_1 = Lambda(lambda x: tf.split(x, 3, axis=3))(main_path)
    block_1 = Conv2D(filters=main_path.shape[-1] // 3, kernel_size=1, strides=1, padding='same', activation='relu')(block_1)
    block_1 = Concatenate(axis=3)(block_1)

    # Block 2: Channel shuffle
    block_2 = Lambda(lambda x: tf.transpose(x, perm=(0, 3, 1, 2)))(block_1)
    block_2 = Conv2D(filters=block_2.shape[-1], kernel_size=3, strides=1, padding='same', activation='relu')(block_2)
    block_2 = Lambda(lambda x: tf.transpose(x, perm=(0, 2, 3, 1)))(block_2)
    block_2 = Conv2D(filters=block_2.shape[-1], kernel_size=3, strides=1, padding='same', activation='relu')(block_2)
    block_2 = Lambda(lambda x: tf.transpose(x, perm=(0, 2, 3, 1)))(block_2)

    # Block 3: Depthwise separable convolution
    block_3 = Conv2D(filters=block_2.shape[-1], kernel_size=3, strides=1, padding='same', activation='relu')(block_2)
    block_3 = Conv2D(filters=block_3.shape[-1], kernel_size=3, strides=1, padding='same', activation='relu')(block_3)

    # Branch
    branch = Conv2D(filters=block_1.shape[-1], kernel_size=1, strides=1, padding='same', activation='relu')(main_path)

    # Combine main path and branch
    combined = tf.concat([block_1, block_2, block_3], axis=3)
    combined = combined + branch

    # Flatten and fully connected layers
    flattened = Flatten()(combined)
    dense = Dense(units=128, activation='relu')(flattened)
    dense = Dense(units=64, activation='relu')(dense)
    output = Dense(units=10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=main_path, outputs=output)

    return model