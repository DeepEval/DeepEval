import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def path_1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def path_2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
        return conv

    def path_3(input_tensor):
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
        conv_3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
        concat = Concatenate()([conv_1x3, conv_3x1])
        return concat

    def path_4(input_tensor):
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
        conv_1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
        conv_3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
        concat = Concatenate()([conv_1x3, conv_3x1])
        return concat

    p1 = path_1(input_layer)
    p2 = path_2(input_layer)
    p3 = path_3(input_layer)
    p4 = path_4(input_layer)

    concat_features = Concatenate()([p1, p2, p3, p4])
    flatten_layer = Flatten()(concat_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model