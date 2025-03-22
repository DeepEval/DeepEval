import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))  

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)

    conv_shortcut = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    added_output = Add()([conv3, conv_shortcut]) 

    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2)(added_output)
    flatten_layer = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model