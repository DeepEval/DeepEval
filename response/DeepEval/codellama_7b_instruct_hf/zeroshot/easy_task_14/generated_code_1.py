import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten
from keras.applications.vgg16 import VGG16


def dl_model():

    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    input_layer = Input(shape=input_shape)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(input_layer)

    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Flatten the output
    x = Flatten()(x)

    # Final Dense layer
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model