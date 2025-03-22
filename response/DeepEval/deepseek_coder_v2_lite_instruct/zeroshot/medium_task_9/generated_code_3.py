import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Branch path (identity)
    branch = Conv2D(filters, (1, 1), padding='same')(input_tensor)
    
    # Feature fusion
    x = Add()([x, branch])
    x = ReLU()(x)
    
    return x

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(16, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First basic block
    x = basic_block(x, 16)
    
    # Second basic block
    x = basic_block(x, 16)
    
    # Average pooling layer
    x = AveragePooling2D((8, 8))(x)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=input_tensor, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()