import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the feature extraction for each group
    def depthwise_conv_block(input_tensor, kernel_size):
        x = Conv2D(filters=None, kernel_size=kernel_size, padding='same', activation='relu', depthwise_mode=True)(input_tensor)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        return x

    # Apply depthwise separable convolutional layers with different kernel sizes
    conv1 = depthwise_conv_block(split_layer[0], kernel_size=(1, 1))
    conv2 = depthwise_conv_block(split_layer[1], kernel_size=(3, 3))
    conv3 = depthwise_conv_block(split_layer[2], kernel_size=(5, 5))

    # Concatenate the outputs of the three groups
    concatenated_output = Concatenate()([conv1, conv2, conv3])

    # Flatten the fused features
    flattened_output = Flatten()(concatenated_output)

    # Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model