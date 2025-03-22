import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # Second path: Parallel branches
    path2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path2_1x3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path2_1x1)
    path2_3x1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path2_1x1)
    
    # Concatenate the outputs of the second path
    concatenated_path2 = Concatenate()([path2_1x1, path2_1x3, path2_3x1])
    
    # Combine the first and second paths
    combined_path = Add()([path1, concatenated_path2])
    
    # Additional 1x1 convolution
    final_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(combined_path)
    
    # Flatten the output
    flattened = Flatten()(final_path)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()