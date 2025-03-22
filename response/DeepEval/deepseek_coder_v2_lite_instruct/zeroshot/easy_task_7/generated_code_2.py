import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Add

def dl_model():
    # Define the main path
    main_input = Input(shape=(28, 28, 1), name="main_input")
    
    # First convolution and dropout block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(main_input)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    
    # Another convolutional layer to restore the number of channels
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Branch path
    branch_input = Input(shape=(28, 28, 1), name="branch_input")
    
    # Add the branch path to the main path
    combined = Add()([x, branch_input])
    
    # Flatten the output
    combined = Flatten()(combined)
    
    # Fully connected layer
    output = Dense(10, activation='softmax')(combined)
    
    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=output)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()