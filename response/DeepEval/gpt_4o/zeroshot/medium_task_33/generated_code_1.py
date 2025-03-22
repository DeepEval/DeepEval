import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer with shape corresponding to the CIFAR-10 dataset (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three separate channels
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define separable convolutions for each channel with different kernel sizes
    conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_channels[0])
    conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
    conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])

    # Concatenate the outputs from the three convolutional layers
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Define three fully connected (Dense) layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='relu')(dense2)

    # Output layer with 10 units for the 10 classes in CIFAR-10, using softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()