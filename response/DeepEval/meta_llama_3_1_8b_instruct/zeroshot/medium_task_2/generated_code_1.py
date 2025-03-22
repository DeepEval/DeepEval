# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the main path of the model
    main_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(main_input)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Define the branch path of the model
    branch_input = keras.Input(shape=input_shape)
    y = layers.Conv2D(32, (5, 5), activation='relu')(branch_input)
    
    # Combine the features from both paths
    combined = layers.Concatenate()([x, y])
    
    # Flatten the combined features
    flat = layers.Flatten()(combined)
    
    # Define the fully connected layers for classification
    x = layers.Dense(128, activation='relu')(flat)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=outputs)
    
    return model

# Usage
model = dl_model()
model.summary()