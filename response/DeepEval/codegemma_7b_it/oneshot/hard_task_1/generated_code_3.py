import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block1(input_tensor):
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=64, activation='relu')(path1)
        path1 = Dense(units=input_tensor.shape[3], activation='sigmoid')(path1)
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=64, activation='relu')(path2)
        path2 = Dense(units=input_tensor.shape[3], activation='sigmoid')(path2)
        output_tensor = Concatenate()([path1, path2])
        output_tensor = keras.layers.multiply([input_tensor, output_tensor])
        return output_tensor

    block1_output = block1(input_tensor=max_pooling)
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(block1_output)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block2(input_tensor):
        path1 = keras.layers.average_pooling2d(input_tensor)
        path2 = keras.layers.max_pooling2d(input_tensor)
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path1 = Conv2D(filters=input_tensor.shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(path1)
        path2 = Conv2D(filters=input_tensor.shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(path2)
        output_tensor = Concatenate()([path1, path2])
        output_tensor = keras.layers.multiply([input_tensor, output_tensor])
        return output_tensor

    block2_output = block2(input_tensor=max_pooling)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(block2_output)
    conv = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    flatten_layer = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model