import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    dropout1 = Dropout(rate=0.2)(conv1_1)
    dropout2 = Dropout(rate=0.2)(conv1_2)
    dropout3 = Dropout(rate=0.2)(conv1_3)
    dropout4 = Dropout(rate=0.2)(conv1_4)
    dropout5 = Dropout(rate=0.2)(conv1_5)

    concat = Add()([dropout1, dropout2, dropout3, dropout4, dropout5])
    flatten_layer = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model