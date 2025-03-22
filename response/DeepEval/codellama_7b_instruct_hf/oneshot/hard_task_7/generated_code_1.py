import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer with 32 kernels
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Splitting the input into two groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv)

    # First group
    first_group = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    first_group = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_group)
    first_group = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_group)

    # Second group
    second_group = split_layer[1]

    # Merging the outputs from both groups
    merged_output = Concatenate()([first_group, second_group])

    # Block 2
    block_2_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged_output)
    block_2_output = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2_output)
    block_2_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_2_output)

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(block_2_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model