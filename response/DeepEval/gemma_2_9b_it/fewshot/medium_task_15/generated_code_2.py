import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Concatenate, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_layer = BatchNormalization()(conv_layer)
    relu_layer = Activation('relu')(bn_layer)
    
    avg_pool_layer = GlobalAveragePooling2D()(relu_layer)
    dense1 = Dense(units=32, activation='relu')(avg_pool_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    reshape_layer = Reshape(target_shape=(32, 1, 1))(dense2)
    weighted_features = Multiply()([reshape_layer, relu_layer])
    
    concatenated_features = Concatenate()([input_layer, weighted_features])
    
    conv_down = Conv2D(filters=16, kernel_size=(1, 1))(concatenated_features)
    avg_pool_down = AveragePooling2D(pool_size=(2, 2))(conv_down)
    
    output_layer = Dense(units=10, activation='softmax')(avg_pool_down)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model