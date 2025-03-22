import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import backend as K

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):

        # Split input into three groups along the last dimension
        input_groups = tf.split(input_tensor, num_or_size_splits=3, axis=-1)

        # Extract deep features from each group
        outputs = []
        for group in input_groups:
            conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
            conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
            outputs.append(conv3)

        # Concatenate outputs from all groups
        output_tensor = Concatenate()([*outputs])

        return output_tensor

    # Block 1
    block1_output = block1(input_tensor=input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block2(input_tensor):

        # Generate channel-matching weights
        fc1 = Dense(units=128, activation='relu')(input_tensor)
        fc2 = Dense(units=128, activation='relu')(fc1)
        weights = K.reshape(fc2, (128, 1, 1, 1))

        # Reshape weights to match input
        weights = K.repeat_elements(weights, repeats=128, axis=-1)
        weights = K.repeat_elements(weights, repeats=128, axis=-2)

        # Multiply adjusted output with weights
        main_path = K.multiply([input_tensor, weights])

        # Direct branch
        branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch)
        branch = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch)

        # Add outputs from both paths
        output = K.add([main_path, branch])

        # Global max pooling
        output = MaxPooling2D(pool_size=(2, 2))(output)

        return output

    block2_output = block2(input_tensor=transition_conv)

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model