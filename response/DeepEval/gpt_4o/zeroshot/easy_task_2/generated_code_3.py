from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Softmax
from keras.models import Model

def dl_model():
    # Define the input layer with the specified shape
    input_layer = Input(shape=(224, 224, 3))
    
    # First sequential feature extraction block
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Second sequential feature extraction block
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Additional convolutional layers
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
    
    # Average pooling layer to reduce dimensionality
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # First fully connected layer with dropout
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    
    # Second fully connected layer with dropout
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    
    # Output layer with softmax activation for classification
    output_layer = Dense(units=1000, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()