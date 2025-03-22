import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def conv_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Apply convolution blocks to each split
    block_outputs = [conv_block(split_input) for split_input in split_inputs]
    
    # Concatenate outputs from the three blocks
    concatenated_output = Concatenate()(block_outputs)

    # Transition Convolution to adjust channels
    transition_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_output)

    # Block 2
    global_pooling = GlobalMaxPooling2D()(transition_conv)
    
    # Fully connected layers to generate weights
    dense1 = Dense(units=32, activation='relu')(global_pooling)
    dense2 = Dense(units=transition_conv.shape[-1], activation='sigmoid')(dense1)  # Match number of channels

    # Reshape the weights
    reshaped_weights = Reshape((1, 1, transition_conv.shape[-1]))(dense2)

    # Multiply the weights with the transition conv output
    weighted_output = Multiply()([transition_conv, reshaped_weights])

    # Branching output directly from the input
    direct_branch = input_layer

    # Add the outputs from the main path and the branch
    final_output = Add()([weighted_output, direct_branch])

    # Fully connected layer for classification
    flatten_output = Flatten()(final_output)
    classification_output = Dense(units=10, activation='softmax')(flatten_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=classification_output)

    return model