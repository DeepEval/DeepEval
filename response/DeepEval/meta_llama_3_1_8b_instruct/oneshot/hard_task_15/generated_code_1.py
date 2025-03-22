import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Add, Dense, concatenate

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    global_avg_pool = GlobalAveragePooling2D()(conv)
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    weights = Dense(units=64)(dense1)
    weights = Reshape((1, 1, 64))(weights)
    
    # Branch Path
    branch = input_layer
    
    # Combine Main and Branch Paths
    addition = Add()([conv, weights, branch])
    
    # Fully Connected Layers
    concat = concatenate([addition, conv])
    dense2 = Dense(units=64, activation='relu')(concat)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model