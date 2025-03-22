import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def conv_block(input_tensor):
        # Apply a series of convolutions: 1x1, 3x3, and another 1x1
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each group through the convolutional block
    outputs = [conv_block(group) for group in inputs_groups]

    # Combine the outputs of the three groups using addition
    main_path = Add()(outputs)

    # Fuse the main path with the original input
    combined = Add()([main_path, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model