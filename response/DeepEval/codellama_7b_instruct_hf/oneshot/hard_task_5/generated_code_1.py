import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    # Process each group with a 1x1 convolution
    conv_layer1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer)
    conv_layer2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer)
    conv_layer3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer)
    # Concatenate the outputs from the three groups
    concatenated_layer = Concatenate()([conv_layer1, conv_layer2, conv_layer3])

    # Block 2
    # Reshape the output from Block 1
    reshaped_layer = Lambda(lambda x: tf.reshape(x, (32, 32, 1, 1)))(concatenated_layer)
    # Swap the third and fourth dimensions
    swapped_layer = Lambda(lambda x: tf.transpose(x, (0, 1, 3, 2)))(reshaped_layer)
    # Reshape the output back to its original shape
    final_layer = Lambda(lambda x: tf.reshape(x, (32, 32, 1, 1)))(swapped_layer)

    # Block 3
    # Apply a 3x3 depthwise separable convolution
    depthwise_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False, depthwise_initializer='glorot_uniform')(final_layer)

    # Branch
    branch_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    branch_conv_layer1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_layer)
    branch_conv_layer2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_layer)
    branch_conv_layer3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_layer)
    branch_concatenated_layer = Concatenate()([branch_conv_layer1, branch_conv_layer2, branch_conv_layer3])

    # Combine the main path and the branch
    combined_layer = Lambda(lambda x: tf.concat([x[0], x[1]], axis=3))([final_layer, branch_concatenated_layer])

    # Fully connected layer
    fc_layer = Dense(units=10, activation='softmax')(combined_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=fc_layer)

    return model