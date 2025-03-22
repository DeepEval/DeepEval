import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    block1 = MaxPooling2D((2, 2))(block1)

    # Split the input into two groups
    block1 = Lambda(lambda x: tf.split(x, 2, axis=-1))(block1)
    block1 = Concatenate()([
        Conv2D(64, (1, 1), activation='relu')(block1[0]),
        Conv2D(64, (3, 3), activation='relu')(block1[1])
    ])

    # Define the second block
    block2 = Conv2D(64, (1, 1), activation='relu')(block1)
    block2 = Conv2D(64, (3, 3), activation='relu')(block2)
    block2 = MaxPooling2D((2, 2))(block2)

    # Merge the outputs of the two blocks
    merged = Concatenate()([block1, block2])

    # Add batch normalization and flatten the output
    merged = BatchNormalization()(merged)
    merged = Flatten()(merged)

    # Add a fully connected layer
    merged = Dense(128, activation='relu')(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dense(10, activation='softmax')(merged)

    # Define the model
    model = Model(inputs=input_layer, outputs=merged)

    return model