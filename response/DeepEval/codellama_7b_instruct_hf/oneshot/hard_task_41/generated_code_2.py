import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_input = input_layer
    block1_output = []
    for pool_size in [1, 2, 4]:
        block1_output.append(
            MaxPooling2D(pool_size=(pool_size, pool_size), strides=pool_size, padding='valid')(block1_input)
        )
    block1_output = Concatenate()(block1_output)
    block1_output = Flatten()(block1_output)
    block1_output = Dense(units=128, activation='relu')(block1_output)
    block1_output = Dense(units=64, activation='relu')(block1_output)
    block1_output = Dropout(0.5)(block1_output)

    # Block 2
    block2_input = block1_output
    block2_output = []
    for branch in [1, 3, 3]:
        block2_output.append(
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_input)
        )
        block2_output.append(
            Conv2D(filters=64, kernel_size=(branch, branch), strides=(1, 1), padding='same', activation='relu')(block2_input)
        )
        block2_output.append(
            MaxPooling2D(pool_size=(branch, branch), strides=branch, padding='same')(block2_input)
        )
    block2_output = Concatenate()(block2_output)
    block2_output = Flatten()(block2_output)
    block2_output = Dense(units=128, activation='relu')(block2_output)
    block2_output = Dense(units=64, activation='relu')(block2_output)
    block2_output = Dropout(0.5)(block2_output)

    # Final output
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model