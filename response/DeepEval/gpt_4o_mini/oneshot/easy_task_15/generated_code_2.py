import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1_2)
        dropout = Dropout(rate=0.5)(avg_pool)
        
        return dropout
    
    block_output_1 = specialized_block(input_tensor=input_layer)
    block_output_2 = specialized_block(input_tensor=block_output_1)
    
    global_avg_pool = GlobalAveragePooling2D()(block_output_2)
    flatten_layer = Flatten()(global_avg_pool)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=dense_layer)

    return model