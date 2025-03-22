import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    batch_norm1 = BatchNormalization()(block1)
    block1_output = Concatenate()([conv, block1, batch_norm1])
    
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    batch_norm2 = BatchNormalization()(block2)
    block2_output = Concatenate()([block1_output, block2, batch_norm2])
    
    block3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    batch_norm3 = BatchNormalization()(block3)
    block3_output = Concatenate()([block2_output, block3, batch_norm3])
    
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model