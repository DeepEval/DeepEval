import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Second Block
    global_avg_pool = GlobalAveragePooling2D()(avg_pool)
    dense1 = Dense(units=16, activation='relu')(global_avg_pool)
    dense2 = Dense(units=16, activation='relu')(dense1)
    reshape_weights = Reshape((32, 32, 16))(dense2)  # Reshape for element-wise multiplication

    weighted_input = Multiply()([avg_pool, reshape_weights]) 

    flatten = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model