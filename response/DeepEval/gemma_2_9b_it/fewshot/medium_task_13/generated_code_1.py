import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_concat = Concatenate(axis=3)([input_layer, conv1])

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_concat)
    conv2_concat = Concatenate(axis=3)([conv1_concat, conv2])

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_concat)

    flatten_layer = Flatten()(conv3)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model