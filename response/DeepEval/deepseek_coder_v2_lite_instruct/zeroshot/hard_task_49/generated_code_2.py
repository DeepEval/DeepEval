import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Lambda, DepthwiseConv2D, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    x = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=4)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)  # Fully connected layer after flattening

    # Reshape the output of the first block to a 4-dimensional tensor
    x = Reshape((4, 4, 8))(x)  # Assuming the output is 8 channels

    # Second block
    # Split the input into four groups along the last dimension
    x = Lambda(lambda tensor: tf.split(tensor, 4, axis=-1))(x)

    # Process each group with depthwise separable convolutional layers
    outputs = []
    for i in range(4):
        group = x[i]
        if i == 0:
            conv = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(group)
        elif i == 1:
            conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(group)
        elif i == 2:
            conv = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(group)
        elif i == 3:
            conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(group)
        conv = Flatten()(conv)
        outputs.append(conv)

    # Concatenate the outputs from all groups
    x = Concatenate()(outputs)

    # Fully connected layer for classification
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    return model

# Example usage
model = dl_model()
model.summary()