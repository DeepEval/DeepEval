import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path = Conv2D(64, (1, 1), activation='relu')(input_layer)  # 1x1 convolution for dimensionality reduction
    main_path_1 = Conv2D(32, (1, 1), activation='relu')(main_path)  # 1x1 convolution for more reduction
    main_path_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path_1)  # 3x3 convolution
    
    # Branch path
    branch_path = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)  # 3x3 convolution
    
    # Concatenate outputs from main and branch paths
    combined = Add()([main_path_2, branch_path])
    
    # Flatten the output
    flattened = Flatten()(combined)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for classification
    
    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()