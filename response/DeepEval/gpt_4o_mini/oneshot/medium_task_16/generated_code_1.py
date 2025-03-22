import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Split the input into 3 groups along the channel dimension
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group
    conv_outputs = []
    for group in split_groups:
        # Number of kernels is one-third of the input channels (3 channels -> 1 kernel per group)
        conv_output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv_outputs.append(conv_output)

    # Average pooling for each convolution output
    pooled_outputs = [AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv) for conv in conv_outputs]

    # Concatenate the pooled outputs along the channel dimension
    concatenated_output = Concatenate(axis=-1)(pooled_outputs)

    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model