import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def dl_model():
    # Define the input shape (28x28 grayscale images)
    input_layer = Input(shape=(28, 28))
    
    # Convolutional layer with average pooling
    x = Conv2D(32, kernel_size=(5, 5), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    # Additional 1x1 convolutional layer
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten and pass through two fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)  # First fully connected layer
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    x = Dense(64, activation='relu')(x)  # Second fully connected layer
    
    # Output layer for multi-class classification
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()

# Print the model summary
model.summary()