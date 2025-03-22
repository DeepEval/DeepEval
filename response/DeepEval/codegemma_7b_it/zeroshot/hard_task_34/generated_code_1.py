from keras import Input, Model
from keras.layers import Conv2D, ReLU, concatenate, Flatten, Dense, add, SeparableConv2D

def dl_model():
    inputs = Input(shape=(28, 28, 1))

    # Main Path
    x = inputs
    for _ in range(3):
        x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
        x = ReLU()(x)
        x = SeparableConv2D(filters=32, kernel_size=3, padding='same')(x)
        x = concatenate([x, inputs])

    # Branch Path
    y = Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
    y = ReLU()(y)

    # Fusion
    z = add([x, y])
    z = Flatten()(z)

    # Output Layer
    outputs = Dense(units=10, activation='softmax')(z)

    model = Model(inputs=inputs, outputs=outputs)

    return model