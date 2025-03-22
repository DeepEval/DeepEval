import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Branch Path
    branch_input = input_layer
    avg_pool = GlobalAveragePooling2D()(branch_input)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=256, activation='relu')(dense1)
    channel_weights = Reshape((32*32,))(dense2)  # Reshape to match the input spatial dimensions

    weighted_input = Multiply()([branch_input, channel_weights])

    # Combine paths
    combined = keras.layers.Add()([pool, weighted_input])

    # Final Classification Layers
    dense3 = Dense(units=512, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model