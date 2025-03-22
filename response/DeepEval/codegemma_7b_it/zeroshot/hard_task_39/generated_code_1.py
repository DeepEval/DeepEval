from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

def dl_model():
    # Input layer
    input_img = keras.Input(shape=(28, 28, 1))

    # Block 1
    pool_out_1 = layers.MaxPooling2D(2, 2)(input_img)
    pool_out_2 = layers.MaxPooling2D(4, 4)(input_img)
    pool_out_3 = layers.MaxPooling2D(8, 8)(input_img)

    # Flatten pooling outputs and concatenate
    concat_1 = layers.Concatenate(axis=-1)([
        layers.Flatten()(pool_out_1),
        layers.Flatten()(pool_out_2),
        layers.Flatten()(pool_out_3)
    ])

    # Reshape for Block 2
    reshaped = layers.Reshape((-1, 1, 1))(concat_1)

    # Block 2 Branches
    conv_out_1 = layers.Conv2D(8, (1, 1), padding='same')(reshaped)
    conv_out_2 = layers.Conv2D(8, (3, 3), padding='same')(reshaped)
    conv_out_3 = layers.Conv2D(8, (5, 5), padding='same')(reshaped)
    pool_out_4 = layers.MaxPooling2D(2, 2)(reshaped)

    # Concatenate branch outputs
    concat_2 = layers.Concatenate(axis=-1)([
        conv_out_1,
        conv_out_2,
        conv_out_3,
        pool_out_4
    ])

    # Classification layer
    flattened = layers.Flatten()(concat_2)
    output = layers.Dense(10, activation='softmax')(flattened)

    # Model definition
    model = Model(inputs=input_img, outputs=output)

    return model