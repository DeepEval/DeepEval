import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    main_path = LayerNormalization()(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)  # Pointwise 1x1
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)  # Pointwise 1x1

    # Branch path
    branch_path = input_layer  # Directly connect to the input

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model