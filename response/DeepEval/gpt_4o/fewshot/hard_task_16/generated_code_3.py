import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Multiply, Add, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
        
        conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)
        
        conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
        conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3_2)

        output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block_2(input_tensor):
        global_pool = GlobalMaxPooling2D()(input_tensor)
        
        dense1 = Dense(units=64, activation='relu')(global_pool)
        dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Output size matches the input channels
        
        channel_weights = Reshape((1, 1, 3))(dense2)
        weighted_output = Multiply()([input_tensor, channel_weights])
        
        return weighted_output

    block2_output = block_2(input_tensor=transition_conv)

    # Branch path directly from the input
    branch_path = input_layer

    # Combine main path and branch path
    combined_output = Add()([block2_output, branch_path])

    # Final Classification Layer
    flatten_layer = GlobalMaxPooling2D()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model