import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1 = input_layer
    for i in range(3):
        block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(block1)
        block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)

    # Define the second block
    block2 = block1
    for i in range(4):
        block2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
        block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
        block2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block2)
        block2 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(block2)
        block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)

    # Flatten and concatenate the outputs of the blocks
    flat1 = Flatten()(block1)
    flat2 = Flatten()(block2)
    concatenated = Concatenate()([flat1, flat2])

    # Add a fully connected layer and a reshape operation
    dense = Dense(units=128, activation='relu')(concatenated)
    reshaped = Reshape((128, 1))(dense)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=reshaped)

    return model