import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def split_and_conv(input_tensor):
        # Splitting the input tensor into 3 groups
        split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Each group performs a series of convolutions
        path1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
        path1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(path1)

        path2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[1])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(path2)

        path3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[2])
        path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(path3)

        # Concatenate the outputs from the three paths
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block1_output = split_and_conv(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    global_pool = GlobalMaxPooling2D()(transition_conv)

    # Generate channel-matching weights through fully connected layers
    dense1 = Dense(units=64, activation='relu')(global_pool)
    dense2 = Dense(units=transition_conv.shape[-1])(dense1)

    # Reshape weights to match the shape of adjusted output
    reshaped_weights = Reshape((1, 1, transition_conv.shape[-1]))(dense2)

    # Main path output
    main_path_output = tf.multiply(transition_conv, reshaped_weights)

    # Branch directly connected to the input
    branch_output = input_layer

    # Combine main path and branch
    final_output = Add()([main_path_output, branch_output])

    # Fully connected layer for classification
    flatten_output = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model