import keras
from keras.layers import Input, Conv2D, Dropout, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def main_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        drop = Dropout(0.2)(conv)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)
        drop = Dropout(0.2)(conv)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)
        return conv
    
    def branch_path(input_tensor):
        return Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    output_tensor = Add()([main_path_output, branch_path_output])
    
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model