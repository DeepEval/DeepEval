import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    output_tensor = Concatenate()([path1, path2, path3, path4])
    batch_norm = BatchNormalization()(output_tensor)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)

    # Second block
    global_avg_pool = GlobalAveragePooling2D()(conv)
    flatten_layer = Reshape((64,))(global_avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_weights = Dense(units=64, activation='linear')(dense2)
    output_weights = Reshape((1, 1, 64))(output_weights)
    weighted_feature_map = Multiply()([conv, output_weights])
    batch_norm = BatchNormalization()(weighted_feature_map)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model