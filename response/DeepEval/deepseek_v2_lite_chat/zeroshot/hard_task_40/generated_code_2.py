import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape
from keras.layers import AveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization

def dl_model():
    # Define the input shape (assuming grayscale images of 28x28 pixels)
    input_shape = (28, 28, 1)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Block 1: Pooling layers and reshaping
    x = inputs
    x = AveragePooling2D(pool_size=(1, 1))(x)  # 1x1 pool
    x = AveragePooling2D(pool_size=(2, 2))(x)  # 2x2 pool
    x = AveragePooling2D(pool_size=(4, 4))(x)  # 4x4 pool
    x = Flatten()(x)  # Flatten the features
    
    # Block 2: Multi-scale feature extraction
    paths = [
        Conv2D(32, (1, 1), activation='relu')(x),  # Path 1
        Conv2D(32, (3, 3), activation='relu')(x),  # Path 2
        Conv2D(32, (3, 3), activation='relu')(x),  # Path 3
        AveragePooling2D(pool_size=(1, 1))(x),  # Path 4
    ]
    outputs = Concatenate()(paths)  # Concatenate all paths
    
    # Add dropout layers for regularization
    outputs = Dropout(0.5)(outputs)
    
    # Fully connected layers
    outputs = Dense(256, activation='relu')(outputs)
    outputs = Dense(10, activation='softmax')(outputs)  # Output layer for classification
    
    # Model configuration
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()