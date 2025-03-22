import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the main path
    main_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(main_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Define the branch path
    branch_input = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_input)
    y = MaxPooling2D((2, 2))(y)
    
    # Combine the outputs from both paths
    combined = Add()([x, y])
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Project onto a probability distribution across 10 classes
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()