import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, DepthwiseConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    x1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    x2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    x3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)

    # Flatten and concatenate the results from the first block
    x = Flatten()(x1)
    y = Flatten()(x2)
    z = Flatten()(x3)
    concat_layer = Concatenate()([x, y, z])

    # Reshape the concatenated output
    reshape_layer = Reshape((1, 1, 9))(concat_layer)

    # Fully connected layer after reshape
    fc_layer = Dense(128, activation='relu')(reshape_layer)

    # Second block
    # Split the input into four groups along the last dimension
    split_layer = Lambda(lambda tensor: tf.split(tensor, 4, axis=-1))(fc_layer)

    # Process each group with depthwise separable convolutional layers
    outputs = []
    kernel_sizes = [(1, 1), (3, 3), (5, 5), (7, 7)]
    for i, kernel_size in enumerate(kernel_sizes):
        group = split_layer[i]
        depthwise_conv = DepthwiseConv2D(kernel_size, padding='same', depthwise_initializer='he_uniform')(group)
        outputs.append(depthwise_conv)

    # Concatenate the outputs from the depthwise separable convolutional layers
    concat_outputs = Concatenate(axis=-1)(outputs)

    # Flatten the concatenated output
    flattened_output = Flatten()(concat_outputs)

    # Fully connected layer to produce the classification result
    final_fc_layer = Dense(10, activation='softmax')(flattened_output)

    # Build the model
    model = Model(inputs=input_layer, outputs=final_fc_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# model = dl_model()
# model.summary()