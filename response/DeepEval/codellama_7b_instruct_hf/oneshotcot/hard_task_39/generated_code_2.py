import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_input = input_layer
    for pool_size, stride in [(1, 1), (2, 2), (4, 4)]:
        block1_input = MaxPooling2D(pool_size=pool_size, strides=stride, padding='valid')(block1_input)
    block1_output = Flatten()(block1_input)

    # Block 2
    block2_input = Concatenate()([
        Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output),
        Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1_output),
        MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    ])
    block2_output = BatchNormalization()(block2_input)
    block2_output = Flatten()(block2_output)
    block2_output = Dense(units=128, activation='relu')(block2_output)
    block2_output = Dense(units=64, activation='relu')(block2_output)
    block2_output = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model