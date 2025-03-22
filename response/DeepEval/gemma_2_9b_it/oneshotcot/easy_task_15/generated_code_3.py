import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        x = Dropout(0.25)(x)

        x = GlobalAveragePooling2D()(x) 
        x = Flatten()(x) 
        output_layer = Dense(units=10, activation='softmax')(x)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model