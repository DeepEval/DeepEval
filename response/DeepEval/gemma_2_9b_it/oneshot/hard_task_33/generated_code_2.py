import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        x = Add()([input_tensor, x])
        return x

    branch1 = block(input_layer)
    branch2 = block(branch1)
    branch3 = block(branch2)

    concatenated = keras.layers.Concatenate()([branch1, branch2, branch3])
    flatten_layer = Flatten()(concatenated)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model