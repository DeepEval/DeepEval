import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    first_block = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    first_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(first_block)

    # Define the second block
    second_block = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_block)
    second_block = GlobalAveragePooling2D()(second_block)
    second_block = Dense(units=128, activation='relu')(second_block)
    second_block = Dense(units=64, activation='relu')(second_block)
    second_block = Dense(units=10, activation='softmax')(second_block)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=second_block)

    return model