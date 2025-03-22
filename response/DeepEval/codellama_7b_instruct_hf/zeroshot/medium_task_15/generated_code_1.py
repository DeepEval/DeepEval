from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten
from keras.applications.cifar10 import CIFAR10



def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = CIFAR10(input_shape=(32, 32, 3), classes=10)

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Add convolutional layers
    conv_layers = []
    for i in range(2):
        conv_layers.append(Conv2D(32, (3, 3), activation='relu', padding='same'))
        conv_layers.append(BatchNormalization())
        conv_layers.append(ReLU())

    # Add global average pooling layer
    pooling_layer = GlobalAveragePooling2D()

    # Add fully connected layers
    fc_layers = []
    for i in range(2):
        fc_layers.append(Dense(128, activation='relu'))
        fc_layers.append(BatchNormalization())

    # Add flatten layer
    flatten_layer = Flatten()

    # Add final fully connected layer
    final_fc_layer = Dense(10, activation='softmax')

    # Create the model
    model = Model(inputs=input_layer, outputs=final_fc_layer)
    model.add(conv_layers)
    model.add(pooling_layer)
    model.add(fc_layers)
    model.add(flatten_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    return model