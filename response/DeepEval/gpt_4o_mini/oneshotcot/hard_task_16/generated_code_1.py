import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # First group of convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1_2)

    # Second group of convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[1])
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2_2)

    # Third group of convolutions
    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[2])
    conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv3_2)

    # Concatenate outputs from three groups
    block1_output = Concatenate()([conv1_3, conv2_3, conv3_3])

    # Transition Convolution Layer
    transition_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    global_pool = GlobalMaxPooling2D()(transition_conv)

    # Fully connected layers to generate weights
    dense1 = Dense(units=64, activation='relu')(global_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)

    # Reshape weights to match the shape of the transition output
    weights = Reshape((1, 1, 32))(dense2)
    weighted_output = tf.multiply(transition_conv, weights)  # Multiply the transition output by weights

    # Branch directly from input
    branch_output = input_layer

    # Add main path output to branch output
    final_output = Add()([weighted_output, branch_output])

    # Final fully connected layer for classification
    flatten_output = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model