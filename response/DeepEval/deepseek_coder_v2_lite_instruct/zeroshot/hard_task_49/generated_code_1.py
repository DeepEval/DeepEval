import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda, Conv2D, DepthwiseConv2D, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # First block
    x = AveragePooling2D(pool_size=(1, 1), strides=1)(inputs)
    x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=4)(x)
    x = Flatten()(x)

    # Fully connected layer and reshape
    x = Dense(64, activation='relu')(x)
    x = Reshape((4, 4, 4))(x)  # Reshape to 4-dimensional tensor

    # Second block
    x = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=4, axis=-1))(x)

    outputs = []
    for i in range(4):
        group = x[i]
        group = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(group)
        group = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(group)
        group = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(group)
        group = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(group)
        outputs.append(group)

    x = Concatenate(axis=-1)(outputs)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    return model

# Create the model
model = dl_model()
model.summary()