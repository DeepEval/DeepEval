import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)

    # Define the second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)

    # Define the third block
    block3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block3 = BatchNormalization()(block3)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block3)

    # Concatenate the outputs of the blocks along the channel dimension
    block_output = Concatenate(axis=1)([block1, block2, block3])

    # Flatten the output and pass it through two fully connected layers
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model