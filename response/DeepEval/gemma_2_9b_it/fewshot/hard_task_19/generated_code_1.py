import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch path
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=128, activation='relu')(dense1)
    channel_weights = Reshape((32, 32, 128))(dense2)  

    # Element-wise multiplication
    weighted_input = input_layer * channel_weights

    # Merge outputs
    merged = Add()([pool, weighted_input])

    # Final classification layers
    flatten = Flatten()(merged)
    dense3 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model