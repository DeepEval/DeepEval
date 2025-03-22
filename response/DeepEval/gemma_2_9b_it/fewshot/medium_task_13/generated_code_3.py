import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    concat1 = Concatenate(axis=3)([conv1, conv2])

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(concat1)
    concat2 = Concatenate(axis=3)([concat1, conv3])

    flatten_layer = Flatten()(concat2)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model