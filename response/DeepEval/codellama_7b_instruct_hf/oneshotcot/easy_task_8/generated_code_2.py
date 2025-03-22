import keras
from keras.layers import Input, DepthwiseSeparableConv2D, Dropout, Flatten, Dense
from keras.models import Model


def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv1 = DepthwiseSeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)


    drop1 = Dropout(rate=0.2)(conv1)


    conv2 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(drop1)
    drop2 = Dropout(rate=0.2)(conv2)


    flatten = Flatten()(drop2)
    fc = Dense(units=128, activation='relu')(flatten)


    fc2 = Dense(units=64, activation='relu')(fc)
    output_layer = Dense(units=10, activation='softmax')(fc2)


    model = Model(inputs=input_layer, outputs=output_layer)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


    model.evaluate(x_test, y_test)

    return model