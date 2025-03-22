import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Add, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    conv1 = Conv2D(32, (3, 3), activation='relu')
    batch_norm1 = BatchNormalization()
    pool1 = MaxPooling2D((2, 2))

    # Define the second block
    conv2 = Conv2D(64, (3, 3), activation='relu')
    batch_norm2 = BatchNormalization()
    pool2 = MaxPooling2D((2, 2))

    # Define the third block
    conv3 = Conv2D(128, (3, 3), activation='relu')
    batch_norm3 = BatchNormalization()
    pool3 = MaxPooling2D((2, 2))

    # Define the parallel branch
    conv_parallel = Conv2D(128, (3, 3), activation='relu')

    # Define the output paths
    output_paths = [conv1, batch_norm1, pool1, conv2, batch_norm2, pool2, conv3, batch_norm3, pool3]

    # Define the parallel branch output
    parallel_output = conv_parallel(input_shape)

    # Define the addition layer
    add_layer = Add()

    # Define the fully connected layers
    dense1 = Dense(128, activation='relu')
    dense2 = Dense(10, activation='softmax')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the output layer
    output_layer = dense2(dense1(add_layer([output_paths, parallel_output])))

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)


    from keras.datasets import cifar10

    # Load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize the data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the model
    model = dl_model()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model