import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, GlobalMaxPooling2D, Dense, Add
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(x):
        # Split the input into three groups
        split_1 = Lambda(lambda z: z[:, :, :, :z.shape[3]//3])(x)
        split_2 = Lambda(lambda z: z[:, :, :, z.shape[3]//3:2*z.shape[3]//3])(x)
        split_3 = Lambda(lambda z: z[:, :, :, 2*z.shape[3]//3:])(x)

        # Process each group through convolutions
        conv_1x1_1 = Conv2D(64, (1, 1), activation='relu')(split_1)
        conv_3x3_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1x1_1)
        conv_1x1_2 = Conv2D(64, (1, 1), activation='relu')(conv_3x3_1)

        conv_1x1_3 = Conv2D(64, (1, 1), activation='relu')(split_2)
        conv_3x3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1x1_3)
        conv_1x1_4 = Conv2D(64, (1, 1), activation='relu')(conv_3x3_2)

        conv_1x1_5 = Conv2D(64, (1, 1), activation='relu')(split_3)
        conv_3x3_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1x1_5)
        conv_1x1_6 = Conv2D(64, (1, 1), activation='relu')(conv_3x3_3)

        # Concatenate the outputs from the three groups
        output = Concatenate()([conv_1x1_2, conv_1x1_4, conv_1x1_6])
        return output

    block1_output = block_1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(32, (1, 1), activation='relu')(block1_output)

    # Block 2
    def block_2(x):
        # Global max pooling
        gp = GlobalMaxPooling2D()(x)
        # Fully connected layers to generate weights
        fc1 = Dense(128, activation='relu')(gp)
        fc2 = Dense(x.shape[3])(fc1)
        # Reshape weights to match the shape of the input
        weights = tf.reshape(fc2, (1, 1, 1, x.shape[3]))
        # Multiply weights with the input
        weighted_features = tf.multiply(x, weights)
        return weighted_features

    block2_output = block_2(transition_conv)

    # Branch connecting directly to the input
    branch = Input(shape=(32, 32, 3))
    branch_output = Conv2D(32, (1, 1), activation='relu')(branch)

    # Addition of main path and branch outputs
    added_output = Add()([block2_output, branch_output])

    # Flatten the result
    flatten_layer = Flatten()(added_output)

    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=[input_layer, branch], outputs=output_layer)

    return model