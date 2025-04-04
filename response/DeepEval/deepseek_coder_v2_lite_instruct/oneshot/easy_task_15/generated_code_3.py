import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3x3)
        dropout = Dropout(0.25)(avg_pool)
        return dropout
    
    block1_output = specialized_block(input_layer)
    block2_output = specialized_block(block1_output)
    
    global_avg_pool = AveragePooling2D(pool_size=(4, 4), strides=1, padding='valid')(block2_output)
    flatten_layer = Flatten()(global_avg_pool)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model