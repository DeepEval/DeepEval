import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    #main_path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    #branch_path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path2)
    
    #concat
    concat = Add()([max_pool2, max_pool4])
    
    #bath_norm
    bath_norm = BatchNormalization()(concat)
    
    #flatten_layer
    flatten_layer = Flatten()(bath_norm)
    
    #dense1
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    #dense2
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    #output_layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model