import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the first block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Define the main path
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    avg_pool_main = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)
    
    # Combine the main path and the first block
    add = Add()([avg_pool_main, avg_pool])
    
    # Define the second block
    gavg_pool = GlobalAveragePooling2D()(add)
    gavg_pool_reshaped = Reshape((64,))(gavg_pool)
    dense1 = Dense(units=64, activation='relu')(gavg_pool_reshaped)
    dense2 = Dense(units=64, activation='relu')(dense1)
    weights = Multiply()([gavg_pool_reshaped, dense2])
    
    # Refine the weights
    weights_reshaped = Reshape((64, 1, 1))(weights)
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weights_reshaped)
    
    # Multiply the weights by the input
    weighted_input = Multiply()([conv5, add])
    
    # Flatten the output and pass it through a fully connected layer for classification
    flatten_layer = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model