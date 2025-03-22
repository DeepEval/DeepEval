import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    
    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(max_pooling)
    
    # Fully connected layer to generate weights
    weights_fc1 = Dense(units=32, activation='relu')(global_avg_pool)
    weights_fc2 = Dense(units=32, activation='relu')(weights_fc1)
    
    # Reshape the weights
    weights_reshape = Reshape((32, 1))(weights_fc2)
    
    # Multiply the input with the learned weights
    element_wise_product = Multiply()([conv, weights_reshape])
    
    # Flatten the output
    flatten_layer = Flatten()(element_wise_product)
    
    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model