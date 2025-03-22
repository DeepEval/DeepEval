import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Softmax
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape of CIFAR-10 images
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolution layer
    x_initial = Conv2D(32, (3, 3), padding='same', activation=None)(inputs)
    
    # Block 1
    x1 = Conv2D(32, (3, 3), padding='same', activation=None)(x_initial)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    # Block 2
    x2 = Conv2D(32, (3, 3), padding='same', activation=None)(x_initial)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    
    # Block 3
    x3 = Conv2D(32, (3, 3), padding='same', activation=None)(x_initial)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    
    # Add the outputs of the blocks to the initial convolution's output
    x = Add()([x_initial, x1, x2, x3])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    
    # Output layer with softmax activation
    outputs = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
# model = dl_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()