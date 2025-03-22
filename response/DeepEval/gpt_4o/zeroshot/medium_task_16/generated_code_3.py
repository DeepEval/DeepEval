from tensorflow.keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Define number of convolutional kernels for each group
    num_kernels = input_shape[-1] // 3

    # Apply 1x1 convolutions and average pooling to each group
    conv_layers = []
    for group in split_layer:
        conv = Conv2D(num_kernels, (1, 1), activation='relu')(group)
        pooled = AveragePooling2D(pool_size=(2, 2))(conv)
        conv_layers.append(pooled)

    # Concatenate the processed groups along the channel dimension
    concatenated = Concatenate(axis=-1)(conv_layers)

    # Flatten the concatenated feature maps
    flattened = Flatten()(concatenated)

    # Pass through two fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)

    # Output layer for classification
    outputs = Dense(num_classes, activation='softmax')(fc2)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
# model = dl_model()
# model.summary()