import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_b = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_a)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1_b)
        dropout = Dropout(rate=0.5)(avg_pool)
        
        return dropout
    
    block1_output = specialized_block(input_tensor=input_layer)
    block2_output = specialized_block(input_tensor=block1_output)
    
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model