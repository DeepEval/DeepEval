import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images to [-1, 1]
    x_train = x_train / 127.5 - 1.
    x_test = x_test / 127.5 - 1.

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        path1 = BatchNormalization()(path1)
        path1 = Activation('relu')(path1)

        # Path 2: Average pooling followed by 1x1 convolution
        path2 = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(path2)
        path2 = BatchNormalization()(path2)
        path2 = Activation('relu')(path2)

        # Path 3: 1x1 convolution followed by two parallel convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1))(input_tensor)
        path3 = BatchNormalization()(path3)
        path3_1x3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path3)
        path3_3x1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path3)
        path3 = Concatenate(axis=-1)([path3, path3_1x3, path3_3x1])
        path3 = BatchNormalization()(path3)
        path3 = Activation('relu')(path3)

        # Path 4: 1x1 convolution followed by a 3x3 convolution
        path4 = Conv2D(filters=32, kernel_size=(1, 1))(input_tensor)
        path4 = BatchNormalization()(path4)
        path4 = Activation('relu')(path4)
        path4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path4)
        path4 = BatchNormalization()(path4)
        path4 = Activation('relu')(path4)

        # Concatenate the outputs of these paths
        multi_scale_output = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Flatten and pass through fully connected layers for classification
        flatten_layer = Flatten()(multi_scale_output)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))

    return model