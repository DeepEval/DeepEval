import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # Define the input shape for CIFAR-10 dataset

    def split_input(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        return inputs_groups

    split_output = split_input(input_tensor=input_layer)

    def process_group(group):
        conv = Conv2D(filters=int(input_layer.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        max_pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv)
        return max_pool

    processed_groups = [process_group(group) for group in split_output]

    concat_output = Concatenate()([group for group in processed_groups])

    flatten_output = Flatten()(concat_output)

    dense1 = Dense(units=128, activation='relu')(flatten_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model