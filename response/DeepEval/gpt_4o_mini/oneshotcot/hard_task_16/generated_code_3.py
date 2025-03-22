import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def split_and_convolve(input_tensor):
        # Split the input tensor into 3 groups along the last dimension
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[2])

        # Concatenate the outputs
        return Concatenate()([conv1, conv2, conv3])

    block1_output = split_and_convolve(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(block1_output)

    # Block 2
    global_pooling = GlobalMaxPooling2D()(transition_conv)

    # Fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)
    channel_weights = Dense(units=transition_conv.shape[-1], activation='sigmoid')(dense2)

    # Reshape the weights to match the output shape of transition convolution
    reshaped_weights = Reshape((1, 1, transition_conv.shape[-1]))(channel_weights)

    # Multiply the adjusted output with the channel weights
    main_path_output = Multiply()([transition_conv, reshaped_weights])

    # Branch output directly from the input
    branch_output = input_layer

    # Add the main path output and branch output
    added_output = Add()([main_path_output, branch_output])

    # Final classification layer
    flatten_output = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model