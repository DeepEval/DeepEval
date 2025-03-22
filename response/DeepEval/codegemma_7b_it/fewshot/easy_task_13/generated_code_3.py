import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv5 = Dropout(0.2)(conv5)
    added = Add()([conv5, input_layer])
    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model