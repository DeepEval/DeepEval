from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten, Reshape, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)

    # First block
    # Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Average pooling layer
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Save the output of the first block
    block1_output = x

    # Second block
    # Global average pooling
    gap = GlobalAveragePooling2D()(block1_output)
    
    # Fully connected layers for channel weighting
    channel_weights = Dense(32, activation='relu')(gap)
    channel_weights = Dense(32, activation='sigmoid')(channel_weights)
    
    # Reshape to match the feature map shape
    channel_weights = Reshape((1, 1, 32))(channel_weights)
    
    # Multiply channel weights with block1_output
    scaled_features = Multiply()([block1_output, channel_weights])
    
    # Flatten and classification layer
    x = Flatten()(scaled_features)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()