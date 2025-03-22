import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1: Splitting input into three paths
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[0])
    path1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path1)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path1)

    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[1])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path2)

    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[2])
    path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path3)

    # Concatenate the outputs from the three paths
    block1_output = Concatenate()([path1, path2, path3])

    # Transition Convolution
    transition_conv = Conv2D(filters=96, kernel_size=(1, 1), padding='same')(block1_output)

    # Block 2: Global Max Pooling and Fully Connected Layers for weights
    global_pooling = GlobalMaxPooling2D()(transition_conv)
    fc1 = Dense(units=128, activation='relu')(global_pooling)
    fc2 = Dense(units=96, activation='sigmoid')(fc1)  # Match the number of channels from transition_conv

    # Reshape weights to match shape of adjusted output and multiply
    weights = Reshape((1, 1, 96))(fc2)
    main_path_output = tf.multiply(transition_conv, weights)

    # Direct branch connection to the input
    branch_output = input_layer

    # Add both paths
    final_output = Add()([main_path_output, branch_output])

    # Fully connected layer for classification
    final_output = GlobalMaxPooling2D()(final_output)  # Global pooling before dense layer
    final_output = Dense(units=10, activation='softmax')(final_output)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model