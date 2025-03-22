from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, MaxPooling2D, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 input shape
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Block 1: Global Average Pooling and Fully Connected Layers
    x1 = GlobalAveragePooling2D()(inputs)
    x1 = Dense(input_shape[-1], activation='relu')(x1)
    x1 = Dense(input_shape[-1], activation='sigmoid')(x1)
    x1 = Reshape((1, 1, input_shape[-1]))(x1)
    weighted_features = Multiply()([inputs, x1])
    
    # Block 2: Convolutional Layers
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(weighted_features)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    
    # Branch connection directly from Block 1 to Block 2 output
    branch = Conv2D(64, (3, 3), activation='relu', padding='same')(weighted_features)
    
    # Fuse outputs from Block 2 and Branch via addition
    fused = Add()([x2, branch])
    
    # Classification layers
    x = Flatten()(fused)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()