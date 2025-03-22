import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Branch 1: Local Feature Extraction
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        # Branch 2: Downsampling & Upsampling
        branch2 = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = UpSampling2D(size=(2, 2))(branch2)

        # Branch 3: Downsampling & Upsampling
        branch3 = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = UpSampling2D(size=(2, 2))(branch3)

        # Concatenate branches
        combined = Concatenate()([branch1, branch2, branch3])
        combined = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined)

        flatten_layer = Flatten()(combined)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer) 

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model