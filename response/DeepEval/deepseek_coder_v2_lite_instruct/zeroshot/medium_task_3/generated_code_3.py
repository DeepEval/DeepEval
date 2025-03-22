import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # First sequential block: Conv2D + MaxPooling2D
    x1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = MaxPooling2D((2, 2))(x1)
    
    # Second sequential block: Conv2D + MaxPooling2D
    x2 = Conv2D(64, (3, 3), activation='relu')(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    
    # Add the outputs from both paths
    combined = Add()([inputs, x2])
    
    # Flatten the combined output
    x = Flatten()(combined)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()