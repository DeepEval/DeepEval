import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)

    # Define the second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)

    # Define the third block
    block3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block3 = BatchNormalization()(block3)
    block3 = ReLU()(block3)

    # Concatenate the outputs of the three blocks
    concat_layer = Concatenate(axis=3)([block1, block2, block3])

    # Flatten the output
    flatten_layer = Flatten()(concat_layer)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model