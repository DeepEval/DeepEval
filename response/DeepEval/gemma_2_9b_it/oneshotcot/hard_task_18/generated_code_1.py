import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Main Path
    global_avg_pool = GlobalAveragePooling2D()(avg_pool)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape for multiplication
    weights = Reshape((32, 32, 32))(dense2)
    
    # Second Block
    
    # Multiply input with channel weights
    output = multiply([avg_pool, weights])

    # Flatten and output layer
    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model