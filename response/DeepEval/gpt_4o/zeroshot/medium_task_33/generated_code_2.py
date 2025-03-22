import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # Split the input into three channel groups
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_channels)(inputs)

    # Feature extraction using separable convolutions
    sep_conv_1x1 = SeparableConv2D(32, (1, 1), activation='relu')(split_layer[0])
    sep_conv_3x3 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split_layer[1])
    sep_conv_5x5 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split_layer[2])

    # Concatenate the outputs from the separable convolutions
    concatenated = Concatenate()([sep_conv_1x1, sep_conv_3x3, sep_conv_5x5])

    # Flatten the concatenated feature maps
    flat = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flat)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(32, activation='relu')(fc2)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(fc3)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Call the function to create the model
model = dl_model()

# Print the model summary
model.summary()