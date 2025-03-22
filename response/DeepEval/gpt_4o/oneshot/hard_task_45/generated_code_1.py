import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: split input into three groups and apply depthwise separable convolutions
    def split_and_convolve(input_tensor):
        # Split into three groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        
        return Concatenate()([conv1x1, conv3x3, conv5x5])

    first_block_output = split_and_convolve(input_layer)

    # Second block with multiple branches
    def multi_branch_block(input_tensor):
        path1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        path2 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        
        path3 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4)
        
        return Concatenate()([path1, path2, path3, path4])

    second_block_output = multi_branch_block(first_block_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model