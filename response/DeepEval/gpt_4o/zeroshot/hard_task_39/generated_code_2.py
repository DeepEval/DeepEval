import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D
from tensorflow.keras.models import Model

def dl_model():
    # Define input
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1 - Multiple scale max pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(inputs)
    
    # Flatten the pool outputs
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    # Concatenate flattened vectors
    concatenated = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer and reshape for Block 2
    fc1 = Dense(units=256, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 16))(fc1)  # Adjust dimensions as needed for your problem
    
    # Block 2 - Multiple branches
    conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    conv3x3 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(reshaped)
    conv5x5 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(reshaped)
    pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    
    # Concatenate outputs from all branches
    concatenated_block2 = Concatenate()([conv1x1, conv3x3, conv5x5, pool3x3])
    
    # Flatten and final fully connected layer for classification
    final_flatten = Flatten()(concatenated_block2)
    outputs = Dense(units=10, activation='softmax')(final_flatten)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of creating and compiling the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()