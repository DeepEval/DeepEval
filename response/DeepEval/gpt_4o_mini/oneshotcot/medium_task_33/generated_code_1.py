import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def feature_extraction(channel_group):
        # Apply separable convolutions of varying sizes
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(channel_group)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(channel_group)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(channel_group)
        return conv1x1, conv3x3, conv5x5

    # Extract features from each channel group
    paths = [feature_extraction(channel) for channel in split_layer]

    # Concatenate the outputs from the three paths
    concatenated = Concatenate()([path for path_group in paths for path in path_group])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model