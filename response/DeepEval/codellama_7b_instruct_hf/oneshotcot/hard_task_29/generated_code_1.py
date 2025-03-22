import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1)

    # Combine main and branch paths
    main_output = Concatenate()([conv1, conv2])
    branch_output = Concatenate()([branch_conv1, branch_conv2])

    # First block
    block1 = Concatenate()([main_output, branch_output])
    block1 = BatchNormalization()(block1)
    block1 = Flatten()(block1)
    block1 = Dense(units=128, activation='relu')(block1)

    # Second block
    block2 = MaxPooling2D(pool_size=(1, 1), strides=1)(block1)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2)(block2)
    block2 = MaxPooling2D(pool_size=(4, 4), strides=4)(block2)
    block2 = Flatten()(block2)
    block2 = Concatenate()([block1, block2])

    # Fully connected layers
    output_layer = Dense(units=10, activation='softmax')(block2)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model