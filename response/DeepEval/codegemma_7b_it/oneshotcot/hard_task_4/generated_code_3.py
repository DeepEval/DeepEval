import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Flatten, Dense, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Conv Layer
    conv_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1)
    
    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(conv_2)
    flatten_avg_pool = Flatten()(avg_pool)
    dense_1 = Dense(units=32, activation='relu')(flatten_avg_pool)
    dense_2 = Dense(units=32, activation='sigmoid')(dense_1)
    reshape_dense_2 = Reshape((1, 1, 32))(dense_2)
    multiply_layer = Multiply()([reshape_dense_2, conv_2])
    
    # Final Conv Layer
    conv_3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(multiply_layer)
    
    # Concatenation
    concat_layer = concatenate([conv_3, input_layer])
    
    # Flattening and Dense Layers
    flatten_layer = Flatten()(concat_layer)
    dense_4 = Dense(units=64, activation='relu')(flatten_layer)
    dense_5 = Dense(units=10, activation='softmax')(dense_4)
    
    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=dense_5)
    
    return model