import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    x = Conv2D(32, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Branch path
    branch = input_tensor
    
    # Addition
    x = Add()([x, branch])
    x = ReLU()(x)
    return x

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    x = Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3))(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First block
    x = basic_block(x)
    branch = Conv2D(16, (1, 1), padding='same')(input_layer)
    x = Add()([x, branch])
    x = ReLU()(x)
    
    # Second block
    x = basic_block(x)
    branch = Conv2D(16, (1, 1), padding='same')(input_layer)
    x = Add()([x, branch])
    x = ReLU()(x)
    
    # Average pooling and flattening
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model