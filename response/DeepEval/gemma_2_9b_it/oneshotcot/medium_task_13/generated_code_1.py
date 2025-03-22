import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional Layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Convolutional Layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Convolutional Layer 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Concatenate outputs
    merged = Concatenate(axis=3)([input_layer, conv1, conv2, conv3])

    # Flatten and Dense Layers
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model