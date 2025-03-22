import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    gap = GlobalAveragePooling2D()(conv)
    dense_gap = Dense(units=32, activation='relu')(gap)
    weights = Dense(units=32*32*3, activation='relu')(dense_gap)
    reshaped_weights = Reshape(target_shape=(32, 32, 3))(weights)
    multiply = keras.layers.Multiply()([reshaped_weights, conv])
    
    # Branch Path
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge Paths
    merge = Add()([multiply, branch])
    
    # Output Layer
    flatten = Flatten()(merge)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model