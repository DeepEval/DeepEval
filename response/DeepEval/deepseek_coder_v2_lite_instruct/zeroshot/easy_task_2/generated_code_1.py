import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(224, 224, 3))

    # First sequential feature extraction layer
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    # Second sequential feature extraction layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    # Additional convolutional layers
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)

    # Another average pooling layer
    x = AveragePooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers with dropout
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    # Output layer with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()