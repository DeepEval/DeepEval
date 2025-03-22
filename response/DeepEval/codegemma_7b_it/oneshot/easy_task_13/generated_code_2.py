import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Dropout(0.2)(conv3)
    conv4 = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv4 = Dropout(0.2)(conv4)
    
    add_layer = Add()([conv1, conv2, conv3, conv4])
    
    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model