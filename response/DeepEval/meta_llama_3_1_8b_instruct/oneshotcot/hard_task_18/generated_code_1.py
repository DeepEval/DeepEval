import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)
    
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    combined_path = Concatenate()([avg_pool, main_path])
    block2_output = combined_path
    
    global_avg_pool = GlobalAveragePooling2D()(main_path)
    channel_weights = Dense(units=64, activation='relu')(global_avg_pool)
    channel_weights = Dense(units=64, activation='relu')(channel_weights)
    
    reshaped_weights = Dense(units=64 * 6 * 6, activation='relu')(channel_weights)
    reshaped_weights = Reshape((64, 6, 6))(reshaped_weights)
    
    multiplied_output = block2_output * reshaped_weights
    
    batch_norm = BatchNormalization()(multiplied_output)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model