import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

        # Combine outputs
        combined = Add()([input_layer, pool2])

        flatten = Flatten()(combined)
        dense = Dense(units=10, activation='softmax')(flatten)

        model = keras.Model(inputs=input_layer, outputs=dense)

        return model