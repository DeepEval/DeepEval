# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # First block: max pooling and dropout
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)  # dropout with a rate of 0.2

    # Reshape the output to a 4D tensor
    x = layers.Reshape((16, 16, 12))(x)  # 16x16x12 because (32/4)x(32/4)x3 = 16x16x3, then 3x12 = 36, but since we're only taking 12, we use 12

    # Second block: separable convolution and concatenation
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(x)
    x1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(x[0])
    x2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(x[1])
    x3 = layers.SeparableConv2D(filters=128, kernel_size=(5, 5), activation='relu')(x[2])
    x4 = layers.SeparableConv2D(filters=256, kernel_size=(7, 7), activation='relu')(x[3])
    x = layers.Concatenate(axis=-1)([x1, x2, x3, x4])

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.summary()