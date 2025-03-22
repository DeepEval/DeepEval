import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    block1 = GlobalAveragePooling2D()(input_shape)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Flatten()(block1)

    # Define the second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    block2 = Concatenate()([block1, block2])
    block2 = BatchNormalization()(block2)
    block2 = Flatten()(block2)

    # Define the third block
    block3 = Dense(units=64, activation='relu')(block2)
    block3 = Dense(units=64, activation='relu')(block3)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(block3)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model