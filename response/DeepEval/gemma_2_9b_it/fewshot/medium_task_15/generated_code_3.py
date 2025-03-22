import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten, Reshape, AveragePooling2D, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn_layer = BatchNormalization()(conv_layer)
    
    pool_layer = GlobalAveragePooling2D()(bn_layer)
    dense1 = Dense(units=32, activation='relu')(pool_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    reshaped_layer = Reshape(target_shape=(32, 32, 32))(dense2)
    weighted_features = reshaped_layer * conv_layer
    
    concatenated_features = Concatenate()([input_layer, weighted_features])
    
    down_sample_layer = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated_features)
    final_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(down_sample_layer)
    
    output_layer = Dense(units=10, activation='softmax')(final_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model