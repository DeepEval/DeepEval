import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, Reshape, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Main Path
    main_path = GlobalAveragePooling2D()(pool)
    
    # Second Block
    dense1 = Dense(units=64, activation='relu')(main_path)
    dense2 = Dense(units=64, activation='relu')(dense1)
    reshape_weights = Reshape(target_shape=(1, 1, 64))(dense2)
    weighted_output = keras.layers.multiply([pool, reshape_weights])
    
    flatten = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model