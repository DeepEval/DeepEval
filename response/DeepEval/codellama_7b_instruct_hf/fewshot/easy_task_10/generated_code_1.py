import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense

 和 return model
def dl_model():
    input_layer = Input(shape=(28, 28, 1))


    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)


    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)


    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(conv2)


    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)


    flatten = Flatten()(maxpool)


    dense1 = Dense(units=128, activation='relu')(flatten)


    dense2 = Dense(units=64, activation='relu')(dense1)


    output_layer = Dense(units=10, activation='softmax')(dense2)


    model = keras.Model(inputs=input_layer, outputs=output_layer)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    return model