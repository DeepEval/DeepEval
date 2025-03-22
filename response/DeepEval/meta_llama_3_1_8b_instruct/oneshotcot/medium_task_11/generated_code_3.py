import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Multiply, BatchNormalization, Flatten, Dense
from keras.layers import AveragePooling2D, MaxPooling2D
from keras import regularizers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def channel_attention(input_tensor):
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=128, activation='relu')(path1)
        path1 = Dense(units=32, activation='sigmoid')(path1)
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=128, activation='relu')(path2)
        path2 = Dense(units=32, activation='sigmoid')(path2)
        output_tensor = Multiply()([path1, path2])
        return output_tensor
        
    attention_output = channel_attention(conv)
    channel_features = Multiply()([conv, attention_output])
    
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(channel_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(channel_features)
    fused_features = Concatenate()([avg_pool, max_pool])
    spatial_features = Multiply()([channel_features, fused_features])

    bath_norm = BatchNormalization()(spatial_features)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model