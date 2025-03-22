import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, kernel_size):
    # Main path
    x1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(input_tensor)
    x2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x1)
    x3 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x2)
    
    # Parallel path directly from input
    parallel = Conv2D(filters, kernel_size, activation='relu', padding='same')(input_tensor)
    
    # Adding the outputs
    combined = Add()([x1, x2, x3, parallel])
    
    return combined

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with a single channel
    inputs = Input(shape=input_shape)
    
    # First block
    block1 = conv_block(inputs, filters=32, kernel_size=(3, 3))
    
    # Second block
    block2 = conv_block(block1, filters=32, kernel_size=(3, 3))
    
    # Concatenation of the two blocks
    concatenated = Concatenate()([block1, block2])
    
    # Final layers
    flatten = Flatten()(concatenated)
    dense = Dense(128, activation='relu')(flatten)
    outputs = Dense(10, activation='softmax')(dense)  # 10 classes for MNIST
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()