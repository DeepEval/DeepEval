import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape (32, 32, 3)

    # Path 1: Two blocks of convolution followed by average pooling
    def path1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv4)

        return pool2

    # Path 2: A single convolutional layer
    def path2(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return conv

    # Apply both paths
    path1_output = path1(input_layer)
    path2_output = path2(input_layer)

    # Combine both paths using addition
    combined_output = Add()([path1_output, path2_output])

    # Flatten the combined output and apply a fully connected layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model