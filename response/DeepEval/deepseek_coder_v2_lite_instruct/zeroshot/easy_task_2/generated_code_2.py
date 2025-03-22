import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Feature extraction layers
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Additional convolutional layers
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # Fully connected layers with dropout
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(units=1000, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()