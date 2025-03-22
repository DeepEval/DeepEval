import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Shortcut path
    shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    
    # Add both paths
    output = Add()([x, shortcut])
    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality to 16 channels
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Apply two consecutive basic blocks
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)
    
    # Average pooling layer
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model