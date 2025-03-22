import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are of size 32x32 with 3 color channels

    # Splitting the input into three groups
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Applying different convolutional kernels to each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

    # Concatenating the outputs from the three convolutional layers
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flattening the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model