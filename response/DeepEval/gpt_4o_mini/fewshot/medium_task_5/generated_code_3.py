import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # First block of convolution and max pooling
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1_main = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_main)

    # Second block of convolution and max pooling
    conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1_main)
    pool2_main = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_main)

    # Branch path
    # Single block of convolution and max pooling
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool_branch = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_branch)

    # Combining both paths using an addition operation
    combined_output = Add()([pool2_main, pool_branch])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model