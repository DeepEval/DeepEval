import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Split, DepthwiseSeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Combine main and branch paths through addition
    combined_path = Concatenate()([main_path, branch_path])

    # Split input into three groups along the channel
    split_input = Lambda(lambda x: tf.split(x, 3, axis=3))(combined_path)

    # Extract features using depthwise separable convolutional layers with different kernel sizes: 1x1, 3x3, and 5x5
    group1 = DepthwiseSeparableConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    group2 = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    group3 = DepthwiseSeparableConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])

    # Concatenate outputs from three groups
    concatenated_output = Concatenate()([group1, group2, group3])

    # Flatten the output and add batch normalization
    flattened_output = Flatten()(concatenated_output)
    flattened_output = BatchNormalization()(flattened_output)

    # Add three fully connected layers to produce classification probabilities
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model