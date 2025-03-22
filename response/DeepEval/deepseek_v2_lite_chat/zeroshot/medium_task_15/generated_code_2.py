from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten, Concatenate
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.merge import concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layer for dimension adjustment
    x = Dense(512)(x)
    x = ReLU()(x)
    
    # Fully connected layer for final output
    x = Dense(10, activation='softmax')(x)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=x)
    
    return model

# Example usage:
model = dl_model()
model.summary()