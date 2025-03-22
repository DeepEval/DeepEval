import keras
import tensorflow as tf
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into 3 channel groups
    channel_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply separable convolutional layers with different kernel sizes to each channel group
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
    conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])

    # Concatenate the outputs from the three separable convolutions
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Pass through fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model