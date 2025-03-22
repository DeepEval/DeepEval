import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Lambda, BatchNormalization, Concatenate, AveragePooling2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: splitting and depthwise separable convolutions
    def split_and_conv(input_tensor):
        # Split the input into three groups
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split2)
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split3)
        
        # Concatenate the results
        output = Concatenate()([conv1, conv2, conv3])
        return output

    block1_output = split_and_conv(input_layer)
    block1_output = BatchNormalization()(block1_output)

    # Second block: multiple branches for feature extraction
    def multi_branch(input_tensor):
        # Branch 1: 1x1 -> 3x3
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

        # Branch 2: 1x1 -> 1x7 -> 7x1 -> 3x3
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch3)

        # Concatenate the results
        output = Concatenate()([branch1, branch2, branch3])
        return output

    block2_output = multi_branch(block1_output)
    block2_output = BatchNormalization()(block2_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model