import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def path1(input_tensor):
        return Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

    def path2(input_tensor):
        x = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        return x

    def path3(input_tensor):
        x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(x1)
        x3 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(x1)
        x = Concatenate()([x2, x3])
        return x

    def path4(input_tensor):
        x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x1)
        x3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(x2)
        x4 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(x2)
        x = Concatenate()([x3, x4])
        return x

    p1 = path1(input_layer)
    p2 = path2(input_layer)
    p3 = path3(input_layer)
    p4 = path4(input_layer)

    concat_layer = Concatenate()([p1, p2, p3, p4])
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model