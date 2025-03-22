import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_img = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_img)

    # Apply 1x1 convolution to the first group
    conv_1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split_channels[0])

    # Apply 3x3 convolution to the second group
    conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split_channels[1])

    # Apply 5x5 convolution to the third group
    conv_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split_channels[2])

    # Concatenate the outputs from the convolution layers
    concatenated = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5])

    # Flatten the concatenated feature maps
    flat = Flatten()(concatenated)

    # Fully connected layer 1
    dense1 = Dense(128, activation='relu')(flat)

    # Fully connected layer 2 (for classification, 10 classes in CIFAR-10)
    output = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_img, outputs=output)

    return model

# Example of creating the model
model = dl_model()
model.summary()