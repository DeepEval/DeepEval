from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, Input, concatenate

def dl_model():

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    x = MaxPooling2D(pool_size=1, strides=1)(inputs)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = MaxPooling2D(pool_size=4, strides=4)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    # Reshape for second block
    x = Dense(4)(x)
    x = Reshape((1, 1, 4))(x)

    # Second block
    groups = 4
    kernel_sizes = (1, 3, 5, 7)
    outputs = []

    for i in range(groups):
        conv = Conv2D(32, (kernel_sizes[i], kernel_sizes[i]), padding='same', use_bias=False,
                      kernel_initializer='he_uniform')(x)
        outputs.append(conv)

    outputs = concatenate(outputs)

    # Classification layer
    x = Flatten()(outputs)
    outputs = Dense(10, activation='softmax')(x)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()