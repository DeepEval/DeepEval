import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting the input into three groups
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def process_group(input_tensor):
        # Sequentially applying convolutions
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Processing each group
    group_outputs = [process_group(group) for group in inputs_groups]

    # Combining the outputs from the three groups using an addition operation
    main_path = Add()(group_outputs)

    # Fusing the main path with the original input layer
    combined_output = Add()([main_path, input_layer])

    # Flattening the combined output into a one-dimensional vector
    flatten_layer = Flatten()(combined_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model