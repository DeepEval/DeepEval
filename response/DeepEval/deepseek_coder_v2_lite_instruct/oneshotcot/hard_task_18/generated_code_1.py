import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Dense, Flatten, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    add1 = Add()([conv2, input_layer])  # Element-wise addition
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(add1)
    
    # Second block
    global_avg_pool = keras.layers.GlobalAveragePooling2D()(avg_pool)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Generate weights from the dense layer
    weights = Dense(units=avg_pool.shape[3], activation='softmax')(dense2)
    
    # Reshape weights to match the shape of the input
    reshaped_weights = keras.layers.Reshape((1, 1, weights.shape[1]))(weights)
    
    # Multiply weights with the input
    weighted_input = Multiply()([avg_pool, reshaped_weights])
    
    # Flatten the weighted input
    flatten_layer = Flatten()(weighted_input)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model