import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)
    drop_out1 = Dropout(0.2)(block1)
    block2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop_out1)
    block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    drop_out2 = Dropout(0.2)(block2)
    global_avg_pool = GlobalAveragePooling2D()(drop_out2)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model