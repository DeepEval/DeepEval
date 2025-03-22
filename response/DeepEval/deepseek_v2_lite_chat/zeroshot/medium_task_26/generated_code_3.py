import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Channel compression
    compressed = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)
    
    # First feature expansion
    expanded1 = Conv2D(filters=64, kernel_size=1, activation='relu')(compressed)
    
    # Second feature expansion
    expanded2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(compressed)
    
    # Concatenate the expanded layers
    concatenated = Concatenate(axis=-1)([expanded1, expanded2])
    
    # Flattening
    flattened = Flatten()(concatenated)
    
    # Fully connected layers
    fc1 = Dense(512, activation='relu')(flattened)
    output = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for simplicity
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Optional: Display the model summary
model = dl_model()
model.summary()