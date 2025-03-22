import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def process_group(input_tensor):
        # Sequential convolutions: 1x1 -> 3x3 -> 1x1
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each group
    processed_groups = [process_group(group) for group in inputs_groups]

    # Combine the outputs of the three groups using addition
    main_path = Add()(processed_groups)

    # Fuse the main path with the original input
    combined_features = Add()([main_path, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(combined_features)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model