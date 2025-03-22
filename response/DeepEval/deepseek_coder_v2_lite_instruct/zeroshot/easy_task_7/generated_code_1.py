import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    
    # Branch path
    y = input_layer
    
    # Combine outputs from both paths
    combined = tf.add(x, y)
    
    # Flatten the combined output
    z = Flatten()(combined)
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(z)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()