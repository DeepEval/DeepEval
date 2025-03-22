import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute

def dl_model():    

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Split the input for Block 1
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=3))(conv1)
    group1_input, group2_input = split_layer

    # Block 1 operations
    group1_conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_input)
    group1_conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(group1_conv1)
    group1_conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_conv2)

    # Group 2 operations
    group2_conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2_input)

    # Concatenate the outputs from both groups
    concat_layer = Concatenate(axis=3)([group1_conv3, group2_conv1])

    # Block 2 operations
    shape_layer = Lambda(lambda x: tf.shape(x))(concat_layer)
    reshape_layer = Reshape([shape_layer[1], shape_layer[2], shape_layer[3] // 2, 2])(concat_layer)
    permute_layer = Permute([1, 2, 4, 3])(reshape_layer)
    reshape_layer2 = Reshape([shape_layer[1], shape_layer[2], shape_layer[3], 2])(permute_layer)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(reshape_layer2)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model