from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer with the shape of 224x224x3
    input_layer = Input(shape=(224, 224, 3))
    
    # First feature extraction layer
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Second feature extraction layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Additional convolutional layers
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    # Average pooling layer to reduce dimensionality
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # First fully connected layer with dropout
    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    
    # Second fully connected layer with dropout
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    
    # Output layer with softmax activation for classification
    output_layer = Dense(units=1000, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model