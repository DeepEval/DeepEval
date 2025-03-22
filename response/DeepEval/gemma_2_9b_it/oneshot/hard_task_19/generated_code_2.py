import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply

def dl_model():     

    input_tensor = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch path
    gap = GlobalAveragePooling2D()(conv3)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=10, activation='sigmoid')(dense1)
    channel_weights = Reshape((32, 32, 10))(dense2)  

    # Multiply channel weights with input
    weighted_input = Multiply()([input_tensor, channel_weights])

    # Combine paths
    combined = keras.layers.add([max_pool, weighted_input]) 
    flatten = Flatten()(combined)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model