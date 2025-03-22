import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)

    # Define the second block
    block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)

    # Concatenate the output of the two blocks
    merged = Concatenate()([block1, block2])

    # Apply batch normalization and flatten the output
    merged = BatchNormalization()(merged)
    merged = Flatten()(merged)

    # Define the final fully connected layers
    output = Dense(units=128, activation='relu')(merged)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model