from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
  
    input_layer = Input(shape=(28,28,1))
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_2)

    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool_1)
    conv_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_4)

    conv_5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool_2)
    conv_6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv_5)
    conv_7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_7)

    conv_8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(pool_3)
    conv_9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv_8)
    conv_10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv_9)
    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_10)

    flatten = Flatten()(pool_4)
    dense_1 = Dense(units=4096, activation='relu')(flatten)
    dense_2 = Dense(units=4096, activation='relu')(dense_1)
    output_layer = Dense(units=1000, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
