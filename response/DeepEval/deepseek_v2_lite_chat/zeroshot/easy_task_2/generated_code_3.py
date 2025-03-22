from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import concatenate

def dl_model():
    # Define the input layer
    input_image = Input(shape=(224, 224, 3))
    
    # First branch: Feature Extraction
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_image)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second branch: Feature Extraction
    branch2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_image)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    
    # Flatten and connect the branches
    branch1 = Flatten()(branch2)
    branch2 = Flatten()(x)
    merged = concatenate([branch1, branch2])
    
    # Additional convolutional layers
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(merged)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Dropout to prevent overfitting
    x = Dropout(0.5)(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output = Dense(1000, activation='softmax')(x)  # Assuming 1000 classes
    
    # Define the model
    model = Model(inputs=input_image, outputs=output)
    
    return model

# Instantiate and return the model
model = dl_model()