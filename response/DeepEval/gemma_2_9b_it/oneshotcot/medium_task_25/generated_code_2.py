import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Path 1
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Path 2
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Path 3
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        path3_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([path3_1, path3_2])

        # Path 4
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        path4_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4_1, path4_2, path4_3])

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        flatten_layer = Flatten()(output_tensor)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model