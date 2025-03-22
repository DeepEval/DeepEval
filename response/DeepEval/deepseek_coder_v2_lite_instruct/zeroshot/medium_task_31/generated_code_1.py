import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Define the different convolutional kernels
    conv1x1 = Conv2D(64, (1, 1), activation='relu')(split_layer[0])
    conv3x3 = Conv2D(64, (3, 3), activation='relu')(split_layer[1])
    conv5x5 = Conv2D(64, (5, 5), activation='relu')(split_layer[2])

    # Concatenate the outputs from the three groups
    concatenated = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Flatten the fused features
    flattened = Flatten()(concatenated)

    # Pass through two fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for CIFAR-10

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()