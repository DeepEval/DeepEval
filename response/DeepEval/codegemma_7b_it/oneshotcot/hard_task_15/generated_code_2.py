import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    gap = GlobalAveragePooling2D()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    weights = Reshape((1, 1, 32))(dense2)
    
    # Branch Path
    branch_output = input_layer
    
    # Combine Paths
    combined = Add()([branch_output, weights])
    
    # Fully Connected Layers
    flatten_layer = Flatten()(combined)
    dense3 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model