import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Conv2DTranspose, BatchNormalization, Activation, add, multiply, concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv_in = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    path1 = GlobalAveragePooling2D()(conv_in)
    path1 = Dense(units=32, activation='relu')(path1)
    path1 = Dense(units=32, activation='relu')(path1)
    
    path2 = GlobalMaxPooling2D()(conv_in)
    path2 = Dense(units=32, activation='relu')(path2)
    path2 = Dense(units=32, activation='relu')(path2)
    
    # Channel attention
    channel_attention = add([path1, path2])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = multiply([conv_in, channel_attention])
    
    # Block 2
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(channel_attention)
    
    concat = concatenate([avg_pool, max_pool], axis=3)
    conv_spatial = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    conv_spatial = Activation('sigmoid')(conv_spatial)
    
    # Spatial attention
    spatial_attention = multiply([channel_attention, conv_spatial])
    
    # Final convolutional layer
    conv_out = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(spatial_attention)
    conv_out = BatchNormalization()(conv_out)
    
    # Final fully connected layer
    flatten = Flatten()(conv_out)
    dense = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model