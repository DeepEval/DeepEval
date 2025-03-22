import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Apply the main path to each group and then combine the results
    main_path_outputs = [main_path(group) for group in split_layer]
    combined_main_path = Add()(main_path_outputs)

    # Add the main path to the original input layer
    added_input = Add()([input_layer, combined_main_path])

    # Flatten the combined features
    flatten_layer = Flatten()(added_input)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model