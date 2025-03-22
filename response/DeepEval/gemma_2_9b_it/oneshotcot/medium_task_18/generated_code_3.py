import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3)) 

    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
    
    concat1 = Concatenate()([conv1_1x1, conv1_3x3, conv1_5x5, pool1])

    conv2_1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv2_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv2_5x5 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concat1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(concat1)

    concat2 = Concatenate()([conv2_1x1, conv2_3x3, conv2_5x5, pool2])

    flatten = Flatten()(concat2)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model