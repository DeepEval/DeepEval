import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))  

    # Path 1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    path1 = x

    # Path 2
    y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = y

    # Concatenate paths
    x = Add()([path1, path2])
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model