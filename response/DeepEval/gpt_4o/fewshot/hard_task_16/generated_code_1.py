import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting input into three groups and applying convolutions
    def block_1(input_tensor):
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv1_2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv1_2)
        
        conv2_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv2_2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv2_2)
        
        conv3_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_groups[2])
        conv3_2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv3_2)
        
        output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])
        return output_tensor

    # Transition Convolution to adjust the number of channels
    def transition_convolution(input_tensor, channels):
        adjusted_output = Conv2D(filters=channels, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return adjusted_output

    # Block 2: Global max pooling, channel-matching weights, and multiplication
    def block_2(input_tensor):
        pooled_output = GlobalMaxPooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(pooled_output)
        channel_weights = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        reshaped_weights = Lambda(lambda x: tf.reshape(x, (-1, 1, 1, x.shape[-1])))(channel_weights)
        weighted_output = Multiply()([input_tensor, reshaped_weights])
        return weighted_output

    block1_output = block_1(input_layer)
    adjusted_output = transition_convolution(block1_output, channels=block1_output.shape[-1])
    block2_output = block_2(adjusted_output)

    # Direct branch from input
    branch_output = input_layer

    # Adding main path output with branch output
    added_output = Add()([block2_output, branch_output])
    
    # Final fully connected layer for classification
    flatten = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model