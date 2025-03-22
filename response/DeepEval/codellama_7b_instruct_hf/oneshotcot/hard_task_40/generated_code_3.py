import keras
from keras.layers import Input, Dense, Flatten, Dropout, Concatenate, AveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Reshape

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    block1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(block1)
    block1 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(block1)
    block1 = Flatten()(block1)

    # Define the second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    block2 = Dropout(0.2)(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    block2 = Dropout(0.2)(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    block2 = Dropout(0.2)(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    block2 = Dropout(0.2)(block2)

    # Concatenate the outputs of the two blocks
    concatenated = Concatenate()([block1, block2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a fully connected layer
    dense = Dense(units=128, activation='relu')(flattened)

    # Add a dropout layer
    dropout = Dropout(0.5)(dense)

    # Add a fully connected layer
    dense2 = Dense(units=64, activation='relu')(dropout)

    # Add a dropout layer
    dropout2 = Dropout(0.5)(dense2)

    # Add a fully connected layer
    output = Dense(units=10, activation='softmax')(dropout2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model