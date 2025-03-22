from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1_2)
    pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(conv1_3)
    drop1 = Dropout(rate=0.5)(pool1)

    conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(drop1)
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2_2)
    pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(conv2_3)
    drop2 = Dropout(rate=0.5)(pool2)

    average_pool = GlobalAveragePooling2D()(drop2)
    flatten = Flatten()(average_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

