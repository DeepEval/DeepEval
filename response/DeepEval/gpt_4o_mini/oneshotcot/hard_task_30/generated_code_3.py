import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First Block
    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch Path
    branch_path = input_layer

    # Combining Paths
    combined_path = Add()([main_path, branch_path])

    # Second Block
    # Splitting the input into three groups along the channel using Lambda layer
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined_path)

    # Extracting features using depthwise separable convolutional layers with different kernel sizes
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])

    # Concatenating the outputs
    concatenated_output = Concatenate()([depthwise_conv1, depthwise_conv2, depthwise_conv3])

    # Flattening the concatenated output
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model