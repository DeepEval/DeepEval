import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_integrate1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    conv_integrate2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_integrate1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_integrate2)
    dropout = Dropout(0.5)(max_pooling)

    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(input_layer)

    fusion = Add()([dropout, conv_branch])

    global_avg_pool = GlobalAveragePooling2D()(fusion)
    flatten_layer = Flatten()(global_avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model