import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)
    block1 = Flatten()(block1)
    block1 = Dense(units=128, activation='relu')(block1)
    block1 = Dropout(rate=0.5)(block1)

    # Define the second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    block2 = Flatten()(block2)
    block2 = Dense(units=128, activation='relu')(block2)
    block2 = Dropout(rate=0.5)(block2)

    # Concatenate the outputs of the two blocks
    concatenated = Concatenate()([block1, block2])

    # Add a fully connected layer and reshape operation
    concatenated = Dense(units=256, activation='relu')(concatenated)
    concatenated = Reshape((256, 1))(concatenated)

    # Define the third block
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenated)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block3)
    block3 = Flatten()(block3)
    block3 = Dense(units=128, activation='relu')(block3)
    block3 = Dropout(rate=0.5)(block3)

    # Define the final output layer
    output = Dense(units=10, activation='softmax')(block3)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model