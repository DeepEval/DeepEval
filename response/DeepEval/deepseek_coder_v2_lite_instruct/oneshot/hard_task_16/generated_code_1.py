import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add, GlobalMaxPooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Split the input into three groups
        split1 = Lambda(lambda tensor: tensor[:, :, :, :tensor.shape[3]//3])(x)
        split2 = Lambda(lambda tensor: tensor[:, :, :, tensor.shape[3]//3:2*tensor.shape[3]//3])(x)
        split3 = Lambda(lambda tensor: tensor[:, :, :, 2*tensor.shape[3]//3:])(x)

        # First group: 1x1 convolution
        conv1 = Conv2D(64, (1, 1), activation='relu')(split1)
        # Second group: 3x3 convolution
        conv2 = Conv2D(64, (3, 3), activation='relu')(split2)
        # Third group: another 1x1 convolution
        conv3 = Conv2D(64, (1, 1), activation='relu')(split3)

        # Concatenate the outputs of the three groups
        output = Concatenate()([conv1, conv2, conv3])
        return output

    # Apply Block 1
    block1_output = block1(input_layer)

    # Transition convolution to adjust the number of channels
    transition_conv = Conv2D(32, (1, 1), activation='relu')(block1_output)

    # Apply Block 2
    block2_output = block2(transition_conv)

    # Add the outputs from both paths
    added_output = Add()([block2_output, input_layer])

    # Flatten the result
    flatten_layer = Flatten()(added_output)

    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def block2(x):
    # Global max pooling
    global_pool = GlobalMaxPooling2D()(x)
    # Two fully connected layers to generate weights
    fc1 = Dense(x.shape[-1] * 4, activation='relu')(global_pool)
    fc2 = Dense(x.shape[-1], activation='relu')(fc1)
    # Reshape weights to match the shape of the input
    weights = tf.reshape(fc2, (1,) + x.shape[1:])
    # Multiply weights with the input to produce the main path output
    main_path_output = tf.multiply(x, weights)
    return main_path_output

# Create the model
model = dl_model()