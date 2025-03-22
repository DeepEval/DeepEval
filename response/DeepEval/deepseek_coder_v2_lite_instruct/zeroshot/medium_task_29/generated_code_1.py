import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    x1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x1)
    
    # Second convolutional layer
    x2 = Conv2D(64, (3, 3), activation='relu')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
    
    # Third convolutional layer
    x3 = Conv2D(128, (3, 3), activation='relu')(x2)
    x3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x3)
    
    # Flatten the output from each pooling layer
    x1_flat = Flatten()(x1)
    x2_flat = Flatten()(x2)
    x3_flat = Flatten()(x3)
    
    # Concatenate the flattened outputs
    combined_features = Concatenate()([x1_flat, x2_flat, x3_flat])
    
    # Fully connected layers
    fc1 = Dense(256, activation='relu')(combined_features)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=inputs, outputs=fc2)
    
    return model

# Example usage
model = dl_model()
model.summary()