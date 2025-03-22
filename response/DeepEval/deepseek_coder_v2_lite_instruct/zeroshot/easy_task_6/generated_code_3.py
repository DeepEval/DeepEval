import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the main path
    main_path_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(main_path_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Define the branch path
    branch_path_input = Input(shape=input_shape)
    branch_path = Conv2D(32, (1, 1), activation='relu')(branch_path_input)
    
    # Combine the two paths
    combined = Add()([x, branch_path])
    
    # Flatten the combined layer
    flattened = Flatten()(combined)
    
    # Fully connected layer
    output = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=[main_path_input, branch_path_input], outputs=output)
    
    return model

# Example usage
model = dl_model()
model.summary()