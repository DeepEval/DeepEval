import keras
from keras.layers import Input, Conv2D, GlobalMaxPooling2D, Dense, Lambda, Add, Multiply
from keras.layers import Concatenate, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input into three groups along the last dimension
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Each group goes through a series of convolutions
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(splits[0])
        path1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path1)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path1)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(splits[1])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path2)

        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(splits[2])
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path3)

        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
    
    block1_output = block1(input_layer)

    # Transition Convolution to adjust channels
    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)

    # Block 2
    def block2(input_tensor, original_input):
        # Global max pooling
        pooled = GlobalMaxPooling2D()(input_tensor)

        # Fully connected layers to generate channel-matching weights
        dense1 = Dense(units=64, activation='relu')(pooled)
        dense2 = Dense(units=3, activation='sigmoid')(dense1)

        # Reshape weights to match the shape of adjusted output
        weights = Reshape((1, 1, 3))(dense2)

        # Multiply weights with the adjusted output
        scaled_output = Multiply()([input_tensor, weights])

        # Add a branch that connects directly to the input
        branch_output = original_input

        # Add outputs from the main path and the branch
        output = Add()([scaled_output, branch_output])

        return output
    
    block2_output = block2(transition_conv, input_layer)

    # Fully connected layer for classification
    flatten_output = tf.keras.layers.Flatten()(block2_output)
    final_output = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model