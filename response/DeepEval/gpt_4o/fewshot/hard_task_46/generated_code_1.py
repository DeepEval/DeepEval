import keras
import tensorflow as tf
from keras.layers import Input, SeparableConv2D, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Channel Splitting and Separable Convolutions
    def block_1(input_tensor):
        channels_split = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channels_split[0])
        sep_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channels_split[1])
        sep_conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channels_split[2])
        
        output_tensor = Concatenate()([sep_conv1, sep_conv2, sep_conv3])
        return output_tensor

    # Second Block: Multiple Branches for Enhanced Feature Extraction
    def block_2(input_tensor):
        # Branch 1: 3x3 Convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max Pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Global Average Pooling and Fully Connected Layer for Classification
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model