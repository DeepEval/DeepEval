import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Convolution operations on each group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 1 processing
    block1_output = block_1(input_tensor=input_layer)

    # Transition convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    global_pooling = GlobalMaxPooling2D()(transition_conv)

    # Two fully connected layers to create channel-matching weights
    dense1 = Dense(units=64, activation='relu')(global_pooling)
    dense2 = Dense(units=transition_conv.shape[-1], activation='sigmoid')(dense1)

    # Reshape weights to match the transition convolution output shape
    reshaped_weights = Reshape(target_shape=(1, 1, transition_conv.shape[-1]))(dense2)

    # Multiply the adjusted output with the weights
    main_path_output = tf.multiply(transition_conv, reshaped_weights)

    # Branch path connecting directly to the input
    branch_output = input_layer

    # Combine the main path output with the branch path
    combined_output = Add()([main_path_output, branch_output])

    # Fully connected layer for classification
    flatten = keras.layers.Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model