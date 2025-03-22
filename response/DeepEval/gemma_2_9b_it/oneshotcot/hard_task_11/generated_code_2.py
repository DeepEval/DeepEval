import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Pathway
    x_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel Branch
    x_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_branch = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(x_branch),
        Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(x_branch)
    ])
    
    # Concatenate and Reduce Dimensions
    x = Concatenate()([x_main, x_branch])
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Direct Connection 
    x = Add()([input_layer, x])  

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model