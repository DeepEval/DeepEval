import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    
    # Input Layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1
    x1 = Conv2D(32, (1, 1))(inputs)

    # Path 2
    x2 = AveragePooling2D((2, 2))(inputs)
    x2 = Conv2D(32, (1, 1))(x2)

    # Path 3
    x3 = Conv2D(32, (1, 1))(inputs)
    x3_1 = Conv2D(32, (1, 3))(x3)
    x3_2 = Conv2D(32, (3, 1))(x3)
    x3 = Concatenate()([x3_1, x3_2])

    # Path 4
    x4 = Conv2D(32, (1, 1))(inputs)
    x4 = Conv2D(32, (3, 3))(x4)
    x4_1 = Conv2D(32, (1, 3))(x4)
    x4_2 = Conv2D(32, (3, 1))(x4)
    x4 = Concatenate()([x4_1, x4_2])

    # Multi-scale feature fusion
    output = Concatenate()([x1, x2, x3, x4])

    # Flatten and classification layer
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    model = Model(inputs=inputs, outputs=output)
    return model