import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, AveragePooling2D, MaxPooling2D, Activation, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel attention path 1: Global Average Pooling
    avg_pool_layer = GlobalAveragePooling2D()(conv_layer)
    dense1 = Dense(units=64, activation='relu')(avg_pool_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Channel attention path 2: Global Max Pooling
    max_pool_layer = GlobalMaxPooling2D()(conv_layer)
    dense3 = Dense(units=64, activation='relu')(max_pool_layer)
    dense4 = Dense(units=32, activation='relu')(dense3)

    # Channel attention fusion
    channel_attention = add([dense2, dense4])
    channel_attention = Activation('sigmoid')(channel_attention) 

    # Element-wise multiplication with original features
    channel_weighted_features = Lambda(lambda x: x[0] * x[1])([conv_layer, channel_attention])

    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_weighted_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_weighted_features)

    # Spatial feature fusion
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Flatten and output layer
    flatten_layer = Flatten()(spatial_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model