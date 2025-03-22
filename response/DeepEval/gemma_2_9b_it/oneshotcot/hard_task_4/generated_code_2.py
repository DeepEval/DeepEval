import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, multiply

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Increase channels
        x = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Initial feature extraction
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                   depth_wise=True)(x)

        # Channel attention module
        channel_attention = GlobalAveragePooling2D()(x)
        channel_attention = Dense(units=16, activation='relu')(channel_attention)
        channel_attention = Dense(units=9, activation='sigmoid')(channel_attention)
        channel_attention = Reshape((32, 32, 9))(channel_attention)
        x = multiply([x, channel_attention])

        # Reduce dimensionality
        x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

        # Combine with input
        x = add([x, input_layer])

        # Flatten and classification
        x = Flatten()(x)
        x = Dense(units=10, activation='softmax')(x)

        model = keras.Model(inputs=input_layer, outputs=x)

        return model