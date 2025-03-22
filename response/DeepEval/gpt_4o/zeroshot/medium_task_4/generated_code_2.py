from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1
    # First convolutional block
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    
    # Second convolutional block
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    
    # Path 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)

    # Combine both paths with addition
    combined = Add()([x1, x2])
    
    # Flatten the combined feature maps
    flattened = Flatten()(combined)
    
    # Fully connected layer to output class probabilities
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()