import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Group 1: 1x1 Convolution
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        # Group 2: 3x3 Convolution
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        # Group 3: 1x1 Convolution
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenate outputs from all groups
        output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block_2(input_tensor):
        global_max_pooling = GlobalMaxPooling2D()(input_tensor)
        # Fully connected layer to generate weights
        weights = Dense(units=64, activation='relu')(global_max_pooling)
        weights = Dense(units=64, activation='sigmoid')(weights)  # We use sigmoid to ensure weights are in [0,1]
        reshaped_weights = Reshape(target_shape=(1, 1, 64))(weights)
        
        # Multiply the adjusted output with the weights
        output_tensor = tf.multiply(input_tensor, reshaped_weights)
        return output_tensor

    block2_output = block_2(input_tensor=transition_conv)

    # Adding main path and branch
    branch_output = transition_conv  # Direct connection to input
    added_output = Add()([block2_output, branch_output])

    # Fully connected layer for classification
    flatten = GlobalMaxPooling2D()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model