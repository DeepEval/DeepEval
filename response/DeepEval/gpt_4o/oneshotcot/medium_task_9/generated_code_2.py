import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
    bn = BatchNormalization()(conv)
    relu = ReLU()(bn)
    
    # Branch path
    branch = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)
    
    # Feature fusion
    output = Add()([relu, branch])
    
    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    reduced_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)
    
    # Two consecutive basic blocks
    block1 = basic_block(reduced_conv, filters=16)
    block2 = basic_block(block1, filters=16)
    
    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2))(block2)
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model