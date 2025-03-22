import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # Convolutional layers in the block
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        
        # Main paths in the branches
        branch1_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)
        branch2_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
        
        # Add the main paths from both branches
        merged_path = Add()([branch1_path, branch2_path])
        
        # Flatten and pass through a fully connected layer
        flattened = Flatten()(merged_path)
        output = Dense(units=10, activation='softmax')(flattened)
        
        return output

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Create two branches with the same block
    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)

    # Concatenate the outputs of the two branches
    concatenated = keras.layers.concatenate([branch1, branch2])

    # Fully connected layer for classification
    output = Dense(units=10, activation='softmax')(concatenated)

    # Model architecture
    model = keras.Model(inputs=input_layer, outputs=output)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Optional: Display model summary
model.summary()