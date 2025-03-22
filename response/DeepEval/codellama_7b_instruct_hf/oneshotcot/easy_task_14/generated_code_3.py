import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import accuracy
from sklearn.preprocessing import LabelEncoder


def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Convert class labels to binary vectors
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Define the model architecture
    input_layer = Input(shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Compile the model with the appropriate loss function and optimizer
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model on the CIFAR-10 dataset
    model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32, validation_data=(X_test, to_categorical(y_test)))

    # Evaluate the model on a test set
    test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
    print('Test accuracy:', test_acc)

    return model