import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16


def dl_model():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


    input_layer = Input(shape=(32, 32, 3))


    path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)


    path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)


    path3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)


    path4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)


    output = Concatenate()([path1, path2, path3, path4])


    output = BatchNormalization()(output)
    output = Flatten()(output)


    output = Dense(units=10, activation='softmax')(output)


    model = keras.Model(inputs=input_layer, outputs=output)


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)