import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, Concatenate
from keras.layers import SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First block: three max pooling layers with different scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs of the pooling layers
    flat1 = Flatten()(max_pool1)
    flat2 = Flatten()(max_pool2)
    flat3 = Flatten()(max_pool3)

    # Apply dropout before concatenation
    drop1 = Dropout(rate=0.5)(flat1)
    drop2 = Dropout(rate=0.5)(flat2)
    drop3 = Dropout(rate=0.5)(flat3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(concatenated)

    # Reshape to prepare for the second block
    reshaped = Reshape((1, 1, 256))(dense1)

    # Second block: split the input into four groups
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each group with separable convolutions
    conv_outputs = []
    kernel_sizes = [1, 3, 5, 7]
    for i, kernel_size in enumerate(kernel_sizes):
        conv = SeparableConv2D(filters=64, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_tensor[i])
        conv_outputs.append(conv)

    # Concatenate the outputs of the convolutions
    concatenated_conv_outputs = Concatenate()(conv_outputs)

    # Flatten the output and pass through a fully connected layer
    flatten_layer = Flatten()(concatenated_conv_outputs)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model