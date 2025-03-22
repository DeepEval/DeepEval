import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, MaxPooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def basic_block(input_tensor):
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = BatchNormalization()(main_path)
        branch_path = input_tensor
        adding_layer = Add()([main_path, branch_path])
        return adding_layer
    
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = basic_block(conv1)
    block2 = basic_block(block1)
    
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block2)
    
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model