import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    Multiply,
    Flatten,
    concatenate,
)
from tensorflow.keras import Model

def block(x, filters):
    avg_pool = GlobalAveragePooling2D()(x)
    fc1 = Dense(filters)(avg_pool)
    fc2 = Dense(filters)(fc1)
    fc2_reshape = Reshape((1, 1, filters))(fc2)
    multiply = Multiply()([fc2_reshape, x])
    return multiply

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Branch 1
    branch_1 = block(input_img, filters=32)

    # Branch 2
    branch_2 = block(input_img, filters=64)

    # Concatenate branches
    concat = concatenate([branch_1, branch_2])

    # Flatten and fully connected layer
    flatten = Flatten()(concat)
    dense = Dense(10, activation="softmax")(flatten)

    # Model definition
    model = Model(inputs=input_img, outputs=dense)

    return model

# Instantiate the model
model = dl_model()