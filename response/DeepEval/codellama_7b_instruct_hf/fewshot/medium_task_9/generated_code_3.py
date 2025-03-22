import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.applications import VGG16


def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    input_layer = Input(shape=input_shape)

    # First block
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    bn1 = BatchNormalization()(conv1)

    # Second block
    conv2 = Conv2D(64, (3, 3), activation='relu')(bn1)
    bn2 = BatchNormalization()(conv2)

    # Add the outputs of both paths
    add = Add()([conv1, conv2])

    # Flatten the output
    flatten = Flatten()(add)

    # Dense layers
    dense1 = Dense(256, activation='relu')(flatten)
    dense2 = Dense(128, activation='relu')(dense1)

    # Output layer
    output = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model