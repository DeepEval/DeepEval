import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Path 1
    def path1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv4)

        return pool2

    # Path 2
    def path2(input_tensor):
        return Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

    # Create paths
    path1_output = path1(input_layer)
    path2_output = path2(input_layer)

    # Merge paths using addition
    merged_output = Add()([path1_output, path2_output])

    # Flatten and fully connected layer
    flattened_output = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)  # 10 classes for CIFAR-10

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model