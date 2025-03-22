import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def path1(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    def path2(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor))

    def path3(input_tensor):
        x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(x1)
        return Concatenate()([x2, x3])
    
    def path4(input_tensor):
        x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(x2)
        x4 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(x2)
        return Concatenate()([x3, x4])

    output_path1 = path1(input_layer)
    output_path2 = path2(input_layer)
    output_path3 = path3(input_layer)
    output_path4 = path4(input_layer)
    output_multi_scale = Concatenate()([output_path1, output_path2, output_path3, output_path4])
    
    flatten_layer = Flatten()(output_multi_scale)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model