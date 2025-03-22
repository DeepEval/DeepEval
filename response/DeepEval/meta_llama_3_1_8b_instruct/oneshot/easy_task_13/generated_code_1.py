import keras
from keras.layers import Input, Conv2D, Add, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_dropout = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_dropout)
    conv2_dropout = Dropout(0.2)(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2_dropout)
    conv3_dropout = Dropout(0.2)(conv3)
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3_dropout)
    conv4_dropout = Dropout(0.2)(conv4)

    conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4_dropout)
    output_tensor = Add()([input_layer, conv5])

    flatten_layer = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model