import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has images of size 32x32 with 3 color channels

    def split_input(input_tensor):
        # Split the input into three groups along the channel dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        return inputs_groups

    def process_group(input_tensor):
        # Apply 1x1 convolutions to each group independently
        conv = Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Downsample via an average pooling layer with consistent parameters
        max_pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv)
        return max_pool

    inputs_groups = split_input(input_layer)
    processed_groups = [process_group(group) for group in inputs_groups]
    concatenated = Concatenate()([group for group in processed_groups])
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model