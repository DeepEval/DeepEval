import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        
        # Block 1
        conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        maxpool1_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_1)

        # Block 2
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        maxpool1_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_2)

        # Combine outputs
        combined = Add()([input_layer, maxpool1_1, maxpool1_2]) 

        # Flatten and classify
        flatten = Flatten()(combined)
        dense = Dense(units=10, activation='softmax')(flatten)

        model = keras.Model(inputs=input_layer, outputs=dense)

        return model