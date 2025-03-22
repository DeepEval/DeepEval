import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    input_shape = (32, 32, 3)

    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)

    # Max-pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Concatenate outputs of different paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path2)
        return Concatenate()([path1, path2, path3])

    block_output = block(pool)
    batch_norm = BatchNormalization()(block_output)
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()