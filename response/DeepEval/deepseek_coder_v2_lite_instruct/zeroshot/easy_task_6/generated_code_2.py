import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the main path
    main_path = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(main_path)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Define the branch path
    branch_path = Input(shape=input_shape)
    
    # Add the main path and branch path
    added = Add()([x, branch_path])
    
    # Flatten the output
    flattened = Flatten()(added)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=[main_path, branch_path], outputs=outputs)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()