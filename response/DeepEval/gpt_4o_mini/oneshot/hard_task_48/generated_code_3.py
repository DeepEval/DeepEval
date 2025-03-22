import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting input and applying separable convolutions
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def separable_conv_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
        return BatchNormalization()(Concatenate()([conv1, conv3, conv5]))

    block1_outputs = [separable_conv_block(tensor) for tensor in split_tensors]
    block1_output = Concatenate()(block1_outputs)

    # Block 2: Four parallel branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)

    path2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(path2)

    path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)
    path3_subpath1 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='same')(path3_conv1)
    path3_subpath2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(path3_conv1)
    path3 = Concatenate()([path3_subpath1, path3_subpath2])

    path4_conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)
    path4_conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path4_conv1)
    path4_subpath1 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='same')(path4_conv2)
    path4_subpath2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(path4_conv2)
    path4 = Concatenate()([path4_subpath1, path4_subpath2])

    # Concatenating all paths from Block 2
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model