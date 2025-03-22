import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.utils import np_utils

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    # Block 1
    block1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    block1 = Conv2D(32, (1, 1), activation='relu')(block1)
    block1 = Conv2D(32, (1, 1), activation='relu')(block1)
    block1 = Conv2D(32, (1, 1), activation='relu')(block1)
    block1 = Concatenate()(block1)

    # Block 2
    block2 = Conv2D(32, (1, 1), activation='relu')(block1)
    block2 = Conv2D(32, (3, 3), activation='relu')(block2)
    block2 = Conv2D(32, (3, 3), activation='relu')(block2)
    block2 = Conv2D(32, (3, 3), activation='relu')(block2)
    block2 = Concatenate()(block2)

    # Block 3
    block3 = Conv2D(32, (1, 1), activation='relu')(block2)
    block3 = Conv2D(32, (3, 3), activation='relu')(block3)
    block3 = Conv2D(32, (3, 3), activation='relu')(block3)
    block3 = Conv2D(32, (3, 3), activation='relu')(block3)
    block3 = Concatenate()(block3)

    # Branch path
    branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch = Flatten()(branch)
    branch = Dense(128, activation='relu')(branch)
    branch = Dense(64, activation='relu')(branch)
    branch = Dense(32, activation='relu')(branch)

    # Concatenate the main path and branch path
    concatenated = Concatenate()([block1, block2, block3, branch])

    # Apply batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten the output
    flattened = Flatten()(batch_norm)

    # Apply fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(32, activation='relu')(fc2)
    output = Dense(10, activation='softmax')(fc3)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model