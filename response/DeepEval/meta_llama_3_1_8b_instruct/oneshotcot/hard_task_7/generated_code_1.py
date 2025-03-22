import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Reshape, Permute, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Block 1
    block_output1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    block_output1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output1)
    block_output1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_output1)

    # Split input into two groups along the last dimension
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=2, axis=-1)

    # Lambda layer to split input
    lambda_layer = Lambda(split_input)(conv)
    
    # First group
    group1 = block_output1
    
    # Second group
    group2 = Lambda(lambda x: x[:, :, :, 1:])(lambda_layer[0])
    
    # Concatenate outputs from both groups
    merged_output = Concatenate()([group1, group2])

    # Block 2
    def get_shape(input_tensor):
        return tf.shape(input_tensor)
    
    def reshape_input(input_tensor):
        shape = get_shape(input_tensor)
        new_shape = (shape[1], shape[2], -1, -1)
        return tf.reshape(input_tensor, new_shape)
    
    def channel_shuffle(input_tensor):
        shape = get_shape(input_tensor)
        perm = [0, 1, 3, 2]
        return tf.transpose(input_tensor, perm)
    
    def restore_original_shape(input_tensor):
        shape = get_shape(input_tensor)
        new_shape = (shape[1], shape[2], -1)
        return tf.reshape(input_tensor, new_shape)
    
    reshaped_input = Lambda(reshape_input)(merged_output)
    channel_shuffled_input = Lambda(channel_shuffle)(reshaped_input)
    output_tensor = Lambda(restore_original_shape)(channel_shuffled_input)
    
    # Batch normalization
    bath_norm = BatchNormalization()(output_tensor)
    
    # Flatten
    flatten_layer = Flatten()(bath_norm)
    
    # Dense layer
    dense = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model