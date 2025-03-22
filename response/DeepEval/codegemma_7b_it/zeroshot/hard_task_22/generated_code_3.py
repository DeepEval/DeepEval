from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv2D, SeparableConv2D, BatchNormalization, Activation, concatenate, add, Dense, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x_main = inputs
    splits = tf.split(x_main, 3, axis=-1)
    output_main = []

    for i in range(3):
        x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(splits[i])
        x = SeparableConv2D(64, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(64, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        output_main.append(x)

    output_main = concatenate(output_main, axis=-1)
    output_main = Conv2D(64, (1, 1), padding='same')(output_main)
    output_main = BatchNormalization()(output_main)
    output_main = Activation('relu')(output_main)
    output_main = Flatten()(output_main)

    # Branch path
    x_branch = Conv2D(64, (1, 1), padding='same')(inputs)
    x_branch = BatchNormalization()(x_branch)
    x_branch = Activation('relu')(x_branch)

    # Fusion
    outputs = add([output_main, x_branch])
    outputs = Activation('relu')(outputs)

    # Output layer
    outputs = Dense(10, activation='softmax')(outputs)

    # Model creation
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model