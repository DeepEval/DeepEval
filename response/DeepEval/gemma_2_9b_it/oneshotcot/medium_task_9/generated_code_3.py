import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))
    
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def basic_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        
        shortcut = input_tensor
        
        return Add()([x, shortcut])

    
    x = basic_block(x)
    x = basic_block(x)
    
    branch_x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Add()([x, branch_x])

    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model