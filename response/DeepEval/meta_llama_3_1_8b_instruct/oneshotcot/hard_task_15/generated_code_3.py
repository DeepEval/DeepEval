import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool = GlobalAveragePooling2D()(conv)
    dense1 = Dense(units=128, activation='relu')(pool)
    weights = Dense(units=3, activation='linear')(dense1)
    weights = Reshape((3, 3))(weights)
    element_wise_mul = Multiply()([conv, weights])
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main and branch paths
    add = Add()([element_wise_mul, branch_path])
    
    # Output layers
    dense2 = Dense(units=128, activation='relu')(add)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model