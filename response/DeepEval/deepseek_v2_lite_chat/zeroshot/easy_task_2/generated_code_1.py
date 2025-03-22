import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

def dl_model():
    # Input shape
    input_shape = (224, 224, 3)

    # Load pre-trained VGG16 but exclude the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Prevent the base model from being trainable
    base_model.trainable = False

    # Your own layers
    # Feature extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(base_model.output)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add more convolutional layers if needed
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = Flatten()(x)

    # Add more hidden layers if needed
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)

    # Output layer
    output = Dense(1000, activation='softmax')(x)

    # Connect the layers
    model = Model(inputs=base_model.input, outputs=output)

    # Print model summary
    model.summary()

    return model

# Build the model
model = dl_model()