import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Add, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Path 1: Two blocks of convolution followed by average pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)

    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1_1)
    pool1_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Path 2: Single convolutional layer
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of both paths using addition
    added = Add()([pool1_2, conv2_1])

    # Flatten the combined output
    flatten_layer = Flatten()(added)

    # Fully connected layer to map to 10 classes
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model