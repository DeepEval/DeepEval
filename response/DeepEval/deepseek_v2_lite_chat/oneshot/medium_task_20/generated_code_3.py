import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the model
    def block(input_tensor):
        # First path: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # Second path: 1x1 -> 3x3 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path1)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        # Third path: 1x1 -> 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        # Fourth path: max pooling -> 1x1 convolution
        path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4)

        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Create the model
    inputs = Input(shape=(32, 32, 3))
    block_output = block(inputs)
    batch_norm = BatchNormalization()(block_output)
    flat = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    outputs = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    return model

# Build the model
model = dl_model()