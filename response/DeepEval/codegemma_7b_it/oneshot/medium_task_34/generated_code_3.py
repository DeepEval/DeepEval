import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling3)
    dropout = Dropout(rate=0.3)(conv4)

    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    upsample1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    upsample1 = Concatenate()([upsample1, conv3])

    upsample2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(upsample1)
    upsample2 = Concatenate()([upsample2, conv2])

    upsample3 = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(upsample2)
    upsample3 = Concatenate()([upsample3, conv1])

    conv6 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(upsample3)

    model = keras.Model(inputs=input_layer, outputs=conv6)

    return model