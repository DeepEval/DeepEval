import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()



def dl_model():
    input_layer = Input(shape=(32, 32, 3))


    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)


    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)


    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)


    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)


    branch_output = Concatenate()([branch1, branch2, branch3])


    bath_norm = BatchNormalization()(branch_output)


    flatten_layer = Flatten()(bath_norm)


    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)


    model = Model(inputs=input_layer, outputs=output_layer)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


    model.evaluate(X_test, y_test)

    return model