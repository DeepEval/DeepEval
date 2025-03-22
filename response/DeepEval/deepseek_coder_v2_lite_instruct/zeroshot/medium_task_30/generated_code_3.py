import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First Average Pooling Layer with 1x1 pooling window and stride
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(inputs)

    # Second Average Pooling Layer with 2x2 pooling window and stride
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(inputs)

    # Third Average Pooling Layer with 4x4 pooling window and stride
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(inputs)

    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # Output layer with 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage:
# model = dl_model()
# model.summary()