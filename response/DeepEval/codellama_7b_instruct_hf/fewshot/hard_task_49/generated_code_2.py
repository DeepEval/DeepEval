import tensorflow as tf
from tensorflow import keras


def dl_model():
    # Define the input layer
    input_layer = keras.layers.Input(shape=(28, 28, 1))

    # Define the first block
    # Process the input through three average pooling layers with varying scales
    maxpool1 = keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = keras.layers.Flatten()(maxpool1)
    maxpool2 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = keras.layers.Flatten()(maxpool2)
    maxpool3 = keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = keras.layers.Flatten()(maxpool3)

    # Concatenate the output of each pooling layer
    concatenated_output = keras.layers.Concatenate()([flatten1, flatten2, flatten3])

    # Define the second block
    # Split the input into four groups along the last dimension
    reshaped_output = tf.reshape(concatenated_output, (batch_size, -1, 4))
    split_output = tf.split(reshaped_output, 4, axis=-1)

    # Process each group through depthwise separable convolutional layers with different kernel sizes
    conv1 = keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    conv2 = keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    conv3 = keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_output[2])
    conv4 = keras.layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_output[3])

    # Concatenate the output of each group
    concatenated_output = keras.layers.Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten the output and pass it through a fully connected layer to produce the classification result
    flattened_output = keras.layers.Flatten()(concatenated_output)
    output_layer = keras.layers.Dense(units=10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model