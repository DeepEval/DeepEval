import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2
    path2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3
    path3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_1 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2]) 

    # Path 4
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate all paths
    output_tensor = Concatenate()([path1, path2, path3, path4])

    # Flatten and output layer
    flatten_layer = Flatten()(output_tensor)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    return model