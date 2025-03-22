import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = Cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define convolutional layers with parallel paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    concat_layer = Concatenate()([path1, path2, path3, path4])

    # Define batch normalization and flatten layers
    bath_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(bath_norm)

    # Define dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model