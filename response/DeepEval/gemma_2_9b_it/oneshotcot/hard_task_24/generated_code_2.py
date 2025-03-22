import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Branch 1
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

        # Branch 2
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
        up2 = UpSampling2D(size=(2, 2))(conv2)

        # Branch 3
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
        up3 = UpSampling2D(size=(2, 2))(conv3)

        # Concatenate branches
        merged = Concatenate()([branch1, up2, up3])

        # Final 1x1 convolution
        conv_merge = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged)

        # Flatten and dense layers
        flatten_layer = Flatten()(conv_merge)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model