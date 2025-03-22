import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block1_output = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block1_output = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    block1_output = Concatenate()([block1_output, block1_output, block1_output])
    block1_output = Dropout(0.2)(block1_output)
    
    # Block 2
    block2_output = Lambda(lambda x: K.concatenate([x[:, :, :, :64], x[:, :, :, 64:128], x[:, :, :, 128:196]], axis=-1))(block1_output)
    block2_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block2_output)
    block2_output = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block2_output)
    block2_output = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block2_output)
    block2_output = Concatenate()([block2_output, block2_output, block2_output])
    block2_output = Dropout(0.3)(block2_output)
    
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model