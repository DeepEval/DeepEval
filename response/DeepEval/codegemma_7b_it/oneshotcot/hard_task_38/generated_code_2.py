import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model(): 
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):
        bn = BatchNormalization()(input_tensor)
        act = Activation('relu')(bn)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(act)
        return conv
    
    path1 = block(max_pooling)
    path2 = block(max_pooling)
    path3 = block(max_pooling)
    
    concat = Concatenate()([path1, path2, path3])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()