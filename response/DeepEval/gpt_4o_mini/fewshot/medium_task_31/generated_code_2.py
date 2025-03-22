import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Split the input into three groups along the channel dimension
    split_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define convolution operations for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_groups[2])

    # Concatenate the outputs from the three convolutional layers
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated output
    flatten = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model