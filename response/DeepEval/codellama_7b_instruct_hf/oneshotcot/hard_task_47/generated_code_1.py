import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    groups = Lambda(tf.split, axis=-1)(input_layer)

    # Define the first block
    block1 = groups[0]
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)

    # Define the second block
    block2 = groups[1]
    block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)

    # Concatenate the outputs from the two blocks
    concatenated = Concatenate()([block1, block2])

    # Flatten the concatenated outputs
    flattened = Flatten()(concatenated)

    # Add a dense layer with 128 units and relu activation
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Add another dense layer with 64 units and relu activation
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Add a final dense layer with 10 units and softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model