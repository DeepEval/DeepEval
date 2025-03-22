import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    conv2_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    dropout_layer1 = Dropout(rate=0.2)(conv3)
    dropout_layer2 = Dropout(rate=0.2)(conv3)

    adding_layer = Add()([dropout_layer1, dropout_layer2])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model