import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Lambda, Dense, Reshape, Permute

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda t: tf.split(t, 3, axis=-1))(inputs)
    x = [Conv2D(int(x[0].shape[-1] / 3), (1, 1), activation='relu')(xi) for xi in x]
    x = Concatenate(axis=-1)(x)

    # Block 2
    shape = x.shape
    x = Reshape((shape[1], shape[2], 3, int(shape[-1] / 3)))(x)
    x = Permute((1, 2, 4, 3))(x)
    x = Reshape(shape)(x)

    # Block 3
    x = Conv2D(int(x.shape[-1]), (3, 3), depthwise_mode=True, activation='relu')(x)

    # Branch path
    branch = AveragePooling2D((8, 8))(x)

    # Concatenate main path and branch path
    combined = Concatenate(axis=-1)([x, branch])

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(combined)

    # Define and return the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
# model = dl_model()
# model.summary()