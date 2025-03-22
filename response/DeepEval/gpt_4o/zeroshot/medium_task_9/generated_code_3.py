from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the basic block
    def basic_block(x, filters):
        # Main path
        conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        bn1 = BatchNormalization()(conv1)
        relu1 = ReLU()(bn1)
        
        # Branch path
        branch_conv = Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
        
        # Feature fusion
        out = Add()([relu1, branch_conv])
        return out

    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality to 16
    initial_conv = Conv2D(16, kernel_size=(3, 3), padding='same')(input_layer)
    
    # First basic block
    x = basic_block(initial_conv, 16)
    
    # Second basic block
    x = basic_block(x, 16)
    
    # Average pooling layer to downsample
    x = AveragePooling2D(pool_size=(8, 8))(x)
    
    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()