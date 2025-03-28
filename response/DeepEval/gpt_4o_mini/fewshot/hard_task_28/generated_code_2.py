import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    norm_layer = LayerNormalization()(main_path)
    pointwise_conv1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(norm_layer)  # Same number of channels as input
    pointwise_conv2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(pointwise_conv1)

    # Branch path
    branch_path = input_layer

    # Combine paths
    combined_output = Add()([pointwise_conv2, branch_path])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model