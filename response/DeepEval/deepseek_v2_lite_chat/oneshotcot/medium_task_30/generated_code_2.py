import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers and max-pooling layers
    conv_layer_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv_layer_1)
    
    conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_1)
    avg_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layer_2)
    
    conv_layer_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_2)
    avg_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layer_3)
    
    # Flatten and concatenate the outputs
    flat_layer = Flatten()(Concatenate()([avg_pool_3, conv_layer_3, conv_layer_2, conv_layer_1]))
    
    # Two fully connected layers
    dense_layer_1 = Dense(units=256, activation='relu')(flat_layer)
    dense_layer_2 = Dense(units=128, activation='relu')(dense_layer_1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer_2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model