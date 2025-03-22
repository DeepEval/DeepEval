from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_shape = (28, 28, 1)

    # Create the three branches
    branch_1 = keras.Sequential([
        layers.Conv2D(32, (1, 1), padding='same', activation='relu'),
        layers.DepthwiseConv2D(3, padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), padding='same', activation='relu')
    ])

    branch_2 = keras.Sequential([
        layers.Conv2D(32, (1, 1), padding='same', activation='relu'),
        layers.DepthwiseConv2D(5, padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), padding='same', activation='relu')
    ])

    branch_3 = keras.Sequential([
        layers.Conv2D(32, (1, 1), padding='same', activation='relu'),
        layers.DepthwiseConv2D(7, padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), padding='same', activation='relu')
    ])

    # Create the input layer
    input_img = keras.Input(shape=input_shape)

    # Apply the three branches to the input
    output_1 = branch_1(input_img)
    output_2 = branch_2(input_img)
    output_3 = branch_3(input_img)

    # Concatenate the outputs from the three branches
    concat = layers.concatenate([output_1, output_2, output_3])

    # Flatten the concatenated output
    flatten = layers.Flatten()(concat)

    # Add a fully connected layer for classification
    output = layers.Dense(10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_img, outputs=output)

    return model