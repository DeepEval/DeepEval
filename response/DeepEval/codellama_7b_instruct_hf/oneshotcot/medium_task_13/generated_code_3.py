import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)


    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)


    def block(input_tensor):
        path1 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(64, (3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(64, (5, 5), activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor


    conv3 = Conv2D(64, (3, 3), activation='relu')(block(max_pooling))
    batch_norm = BatchNormalization()(conv3)


    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(64, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)


    model = Model(inputs=input_layer, outputs=dense2)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    return model