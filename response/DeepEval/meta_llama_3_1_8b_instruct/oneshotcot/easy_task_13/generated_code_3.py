import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    drop1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(drop2)
    drop3 = Dropout(0.2)(conv3)
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(drop3)
    drop4 = Dropout(0.2)(conv4)
    restore = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop4)
    add_layer = Add()([input_layer, restore])
    
    conv_block = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(add_layer)
    drop_block = Dropout(0.2)(conv_block)
    flatten_layer = Flatten()(drop_block)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model