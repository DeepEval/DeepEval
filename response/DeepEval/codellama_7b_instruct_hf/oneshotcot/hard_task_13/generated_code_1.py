import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block2)
    block4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(block3)
    block5 = Concatenate()([block1, block2, block3, block4])

    # Define the second block
    block6 = Flatten()(block5)
    block7 = Dense(units=128, activation='relu')(block6)
    block8 = Dense(units=64, activation='relu')(block7)
    output_layer = Dense(units=10, activation='softmax')(block8)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model