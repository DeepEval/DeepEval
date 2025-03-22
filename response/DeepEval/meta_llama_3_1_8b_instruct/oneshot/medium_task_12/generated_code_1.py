import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras import regularizers

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv3 = BatchNormalization()(conv3)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)
        avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)
        output_tensor = Concatenate()([maxpool, avgpool])
        
        return output_tensor
    
    block_output = block(input_layer)
    block_output = block(block_output)
    block_output = block(block_output)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model