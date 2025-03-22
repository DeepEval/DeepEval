import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    block1_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block2_output = block(input_tensor=block1_max_pool)
    block3_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_output)
    
    block3_output = block(input_tensor=block3_max_pool)
    block4_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block3_output)
    
    bath_norm = BatchNormalization()(block4_max_pool)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model