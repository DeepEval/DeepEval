import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main Path
    initial_conv_main = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv_main)

    # Branch 2
    branch2 = MaxPooling2D(pool_size=(2, 2))(initial_conv_main)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3
    branch3 = MaxPooling2D(pool_size=(4, 4))(initial_conv_main)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(4, 4))(branch3)

    # Concatenate outputs from all branches
    concat_output = Concatenate()([branch1, branch2, branch3])

    # Final convolution in main path
    main_path_output = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(concat_output)

    # Branch Path
    initial_conv_branch = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs from both paths
    added_output = Add()([main_path_output, initial_conv_branch])

    # Fully connected layers for classification
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model