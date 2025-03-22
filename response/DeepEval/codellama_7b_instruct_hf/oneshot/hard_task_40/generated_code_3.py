import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    first_block_output = input_layer
    for i in range(3):
        first_block_output = MaxPooling2D(pool_size=(2 ** (i + 1), 2 ** (i + 1)), strides=2 ** (i + 1), padding='valid')(first_block_output)
    first_block_output = Flatten()(first_block_output)
    first_block_output = Dropout(0.5)(first_block_output)

    # Define the second block
    second_block_output = Concatenate()([
        Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_block_output),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block_output),
        Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(first_block_output),
        Dropout(0.5)(first_block_output)
    ])
    second_block_output = Flatten()(second_block_output)
    second_block_output = Dropout(0.5)(second_block_output)

    # Define the third block
    third_block_output = Concatenate()([
        Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(second_block_output),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block_output),
        Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(second_block_output),
        Dropout(0.5)(second_block_output)
    ])
    third_block_output = Flatten()(third_block_output)
    third_block_output = Dropout(0.5)(third_block_output)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(third_block_output)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model