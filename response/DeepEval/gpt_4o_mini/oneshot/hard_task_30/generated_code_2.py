import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, BatchNormalization
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # First Block
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine both paths
    combined_path = Add()([main_path, branch_path])

    # Second Block
    # Split the combined path into three groups along the channel
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined_path)

    # Depthwise separable convolutions for each split
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])

    # Concatenate the outputs from depthwise convolutions
    concatenated = Concatenate()([depthwise_conv1, depthwise_conv2, depthwise_conv3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model