import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        
        return max_pooling
    
    main_path = block(input_tensor=input_layer)
    main_path = block(main_path)
    
    branch_path = block(input_tensor=input_layer)
    
    merged_path = Add()([main_path, branch_path])
    
    bath_norm = BatchNormalization()(merged_path)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model