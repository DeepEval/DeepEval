import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Reshape, Multiply, Conv2DTranspose, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=8, padding='same')(conv_main)
    
    # Generate weights
    weights = Dense(units=3, activation='linear')(avg_pool)
    weights = Reshape((1, 1, 3))(weights)
    
    # Multiply weights with the original feature map
    multiplied = Multiply()([conv_main, weights])
    
    # Branch path
    branch = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(multiplied)
    
    # Add outputs from both paths
    output = keras.layers.Add()([multiplied, branch])
    
    # Flatten the output
    output = Flatten()(output)
    
    # Apply batch normalization
    output = BatchNormalization()(output)
    
    # Three fully connected layers
    dense1 = Dense(units=128, activation='relu')(output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model