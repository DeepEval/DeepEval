import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split input into three groups along the channel
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path with separable convolutions
    def main_path(split_input):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', use_bias=False)(split_input)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', use_bias=False)(split_input)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', use_bias=False)(split_input)
        return Concatenate()([path1, path2, path3])

    # Apply multi-scale feature extraction to each split
    main_path_outputs = [main_path(split_input) for split_input in split_inputs]
    main_path_output = Concatenate()(main_path_outputs)

    # Branch path with 1x1 convolution
    branch_path = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu', use_bias=False)(input_layer)

    # Fuse outputs from main and branch paths
    combined_output = Add()([main_path_output, branch_path])

    # Flatten and add dense layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model