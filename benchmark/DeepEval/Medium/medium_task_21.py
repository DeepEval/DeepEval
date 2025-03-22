from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    path1 = Dropout(rate=0.5)(path1)

    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv2)
    path2 = Dropout(rate=0.5)(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv3)
    path3 = Dropout(rate=0.5)(conv3)

    pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
    path4 = Dropout(rate=0.5)(conv4)

    concated = Concatenate()([path1, path2, path3, path4])
    dense_1_output = Dense(units=128, activation='relu')(concated)
    dense_2_output = Dense(units=128, activation='relu')(dense_1_output)
    output_layer = Dense(units=10, activation='softmax')(dense_2_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model