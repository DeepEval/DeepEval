import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Reshape, Multiply, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(input_layer)
    weights = Dense(units=3, activation='linear')(avg_pool)
    weights = Reshape((1, 1, 3))(weights)
    multiplied = Multiply()([input_layer, weights])
    main_output = multiplied
    
    # Branch path
    branch_conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    combined = Add()([main_output, branch_conv])
    
    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(batch_norm)
    
    # Three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model