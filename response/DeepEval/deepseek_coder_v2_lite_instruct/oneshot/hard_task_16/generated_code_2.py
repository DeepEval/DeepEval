import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Split the input into three groups
        split1 = Lambda(lambda tensor: tensor[:, :, :, :tensor.shape[3]//3])(x)
        split2 = Lambda(lambda tensor: tensor[:, :, :, tensor.shape[3]//3:2*tensor.shape[3]//3])(x)
        split3 = Lambda(lambda tensor: tensor[:, :, :, 2*tensor.shape[3]//3:])(x)

        # Extract deep features for each group
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1)
        conv3_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_1)
        conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split2)
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split3)
        conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_3)
        conv1_4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_2)

        # Concatenate the outputs
        concatenated = Concatenate()([conv1_2, conv3_1, conv1_4])
        return concatenated

    # Apply Block 1
    block1_output = block1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(block1_output)

    # Apply Block 2
    block2 = Model(inputs=input_layer, outputs=transition_conv)
    block2_output = block2(transition_conv)
    global_max_pool = GlobalMaxPooling2D()(block2_output)

    # Generate channel-matching weights
    fc1 = Dense(units=128, activation='relu')(global_max_pool)
    fc2 = Dense(units=64, activation='relu')(fc1)
    weights = Dense(units=block2_output.shape[3], activation='softmax')(fc2)

    # Reshape weights to match the shape of adjusted output
    reshaped_weights = Lambda(lambda x: tf.reshape(x, (1, 1, 1, -1)))(weights)
    main_path_output = Lambda(lambda x: x[0] * x[1])([reshaped_weights, block2_output])

    # Branch directly from the input
    branch = Input(shape=(32, 32, 3))
    branch_output = block2(branch)

    # Add the outputs from both paths
    added_output = tf.add(main_path_output, branch_output)

    # Fully connected layer for classification
    flattened = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the final model
    model = Model(inputs=[input_layer, branch], outputs=output_layer)

    return model