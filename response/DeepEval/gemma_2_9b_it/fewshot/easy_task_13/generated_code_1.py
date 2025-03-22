import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.25)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    dropout3 = Dropout(rate=0.25)(conv4)

    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(dropout3)

    output = Add()([input_layer, conv5])

    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model