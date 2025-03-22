import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Concatenate, Flatten
from keras.regularizers import l2

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    global_avg_path = GlobalAveragePooling2D()(initial_conv)
    avg_dense1 = Dense(units=64, activation='relu')(global_avg_path)
    avg_dense2 = Dense(units=10, activation='softmax')(avg_dense1)
    
    global_max_path = GlobalMaxPooling2D()(initial_conv)
    max_dense1 = Dense(units=64, activation='relu')(global_max_path)
    max_dense2 = Dense(units=10, activation='softmax')(max_dense1)
    
    channel_attention = Add()([avg_dense2, max_dense2])
    channel_attention = keras.layers.Lambda(lambda x: keras.backend.softmax(x, axis=-1))(channel_attention)
    element_wise_channel = Multiply()([initial_conv, channel_attention])
    
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(element_wise_channel)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(element_wise_channel)
    spatial_features = Concatenate()([avg_pool, max_pool])
    
    channel_features = element_wise_channel
    element_wise = Multiply()([spatial_features, channel_features])
    
    flatten_layer = Flatten()(element_wise)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model