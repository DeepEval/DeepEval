import keras
import tensorflow as tf
from keras.layers import Input, SeparableConv2D, Lambda, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        # Branch 1: Simple 3x3 Conv
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 Conv followed by two 3x3 Convs
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max Pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Connecting the blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Global Average Pooling and Fully Connected Layer
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model