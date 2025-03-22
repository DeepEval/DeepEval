import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Second Block
    global_avg_pool = GlobalAveragePooling2D()(avg_pool)
    fc1 = Dense(units=16, activation='relu')(global_avg_pool)
    fc2 = Dense(units=16, activation='relu')(fc1)
    
    # Reshape and multiply with input
    channel_weights = Reshape((32, 32, 16))(fc2)
    weighted_input = Multiply()([input_layer, channel_weights])

    # Flatten and classify
    flatten_layer = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model