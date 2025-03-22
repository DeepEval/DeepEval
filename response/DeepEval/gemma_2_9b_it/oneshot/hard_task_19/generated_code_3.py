import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 
    
    # Main Path
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch Path
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    channel_weights = Reshape((32 * 32,))(dense2)  

    # Multiplying channel weights with the input
    weighted_input = Multiply()([input_layer, channel_weights]) 

    # Concatenate outputs
    concat_layer = keras.layers.concatenate([max_pool, weighted_input], axis=-1)

    # Final Classification Layers
    dense3 = Dense(units=128, activation='relu')(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model