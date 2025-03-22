import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Multiply, Add, Activation, AveragePooling2D, MaxPooling2D, Concatenate, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Channel Attention Block
    global_avg_pool = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=64, activation='relu')(global_avg_pool)
    dense2_avg = Dense(units=64, activation='sigmoid')(dense1_avg)

    global_max_pool = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=64, activation='relu')(global_max_pool)
    dense2_max = Dense(units=64, activation='sigmoid')(dense1_max)

    # Combining attention weights
    channel_attention = Add()([dense2_avg, dense2_max])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Multiply()([conv, channel_attention])
    
    # Spatial Feature Extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    
    # Fusing features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Combining spatial and channel features
    combined_features = Multiply()([channel_attention, spatial_features])
    
    # Flattening and final output layer
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()