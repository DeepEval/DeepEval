import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the first pathway
    def pathway(input):
        x = input
        for _ in range(3):
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x
    
    # Define the second pathway
    def pathway(input):
        x = input
        for _ in range(3):
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x
    
    # Create the input layer
    inputs = Input(shape=input_shape)
    
    # Apply the pathways
    x1 = pathway(inputs)
    x2 = pathway(inputs)
    
    # Concatenate the outputs of both pathways along the channel dimension
    x = Concatenate()([x1, x2])
    
    # Flatten the concatenated output
    x = Flatten()(x)
    
    # Add two fully connected layers
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])