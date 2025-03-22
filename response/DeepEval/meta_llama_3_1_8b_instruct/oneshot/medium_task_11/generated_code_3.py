import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Dense, concatenate, Flatten, BatchNormalization

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def channel_attention(path, name):
        global_avg_pool = GlobalAveragePooling2D()(path)
        avg_fc1 = Dense(units=128, activation='relu')(global_avg_pool)
        avg_fc2 = Dense(units=128, activation='sigmoid')(avg_fc1)
        
        global_max_pool = GlobalMaxPooling2D()(path)
        max_fc1 = Dense(units=128, activation='relu')(global_max_pool)
        max_fc2 = Dense(units=128, activation='sigmoid')(max_fc1)

        output = Add()([avg_fc2, max_fc2])
        output = Dense(units=128, activation='sigmoid')(output)

        output = Multiply()([output, path])
        return output

    conv1 = channel_attention(conv, 'conv1')

    def spatial_feature_extraction(path):
        avg_pool = AveragePooling2D(pool_size=(8, 8))(path)
        max_pool = MaxPooling2D(pool_size=(8, 8))(path)
        output = concatenate([avg_pool, max_pool], axis=1)
        return output

    spatial_features = spatial_feature_extraction(conv1)

    # Combine channel features with spatial features
    combined_features = Multiply()([conv1, spatial_features])

    bath_norm = BatchNormalization()(combined_features)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model