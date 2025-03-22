import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path3 = Conv2D(32, (1, 1), activation='relu')(path3)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path4 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path4 = Conv2D(32, (3, 3), activation='relu')(path4)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)
    path4 = Conv2D(32, (3, 3), activation='relu')(path4)


    concat = keras.layers.concatenate([path1, path2, path3, path4])
    fc = Dense(10, activation='softmax')(concat)


    model = keras.models.Model(inputs=input_layer, outputs=fc)
    return model