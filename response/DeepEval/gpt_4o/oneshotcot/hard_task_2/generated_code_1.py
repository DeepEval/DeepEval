import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_channels)(input_layer)

    # Process each group through a series of convolutions
    def process_group(group):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(group)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(conv2)
        return conv3

    group1 = process_group(split_layer[0])
    group2 = process_group(split_layer[1])
    group3 = process_group(split_layer[2])

    # Combine the outputs from the three groups
    main_path = Add()([group1, group2, group3])

    # Fuse the main path with the original input
    combined = Add()([main_path, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model