import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Define the input layer with CIFAR-10 shape
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    # Lambda layer to split the input
    split_layer = Lambda(split_input)(input_layer)

    def process_group(group_input):
        # 1x1 Convolution
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(group_input)
        # 3x3 Convolution
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        # Another 1x1 Convolution
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(conv2)
        return conv3

    # Process each group through the same series of convolutions
    group_outputs = [process_group(group) for group in split_layer]

    # Combine the outputs of the three groups with an addition operation
    main_path = Add()(group_outputs)

    # Fuse the main path with the original input layer
    fused_output = Add()([main_path, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(fused_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model