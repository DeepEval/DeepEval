import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three channel groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define separable convolution paths for each channel group
    def separable_conv_block(input_tensor):
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    # Apply the separable convolution block to each channel group
    processed_channels = [separable_conv_block(channel) for channel in split_channels]

    # Concatenate the outputs of all processed channel groups
    concatenated_output = Concatenate()(processed_channels)

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model