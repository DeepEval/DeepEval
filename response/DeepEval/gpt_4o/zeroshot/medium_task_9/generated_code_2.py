from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def basic_block(x, filters, kernel_size=3, stride=1):
    # Main path
    x_main = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x_main = BatchNormalization()(x_main)
    x_main = ReLU()(x_main)
    
    # Branch path
    x_branch = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
    
    # Feature fusion
    x = Add()([x_main, x_branch])
    
    return x

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(input_layer)
    
    # Two consecutive basic blocks
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)
    
    # Average pooling to downsample the feature map
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model