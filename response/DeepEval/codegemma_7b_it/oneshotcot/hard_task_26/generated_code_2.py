import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model(): 

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    
    def main_path(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path3)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path4)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    def branch_path(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = UpSampling2D(size=(2, 2))(path2)
        path3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = UpSampling2D(size=(4, 4))(path3)
        path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = UpSampling2D(size=(8, 8))(path4)
        path5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path5 = UpSampling2D(size=(16, 16))(path5)
        output_tensor = Concatenate()([path1, path2, path3, path4, path5])
        return output_tensor
    
    main_path_output = main_path(input_tensor=max_pooling)
    branch_path_output = branch_path(input_tensor=max_pooling)
    merged = Concatenate()([main_path_output, branch_path_output])
    
    bath_norm = BatchNormalization()(merged)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model