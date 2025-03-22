import keras
from keras.layers import Input, Conv2D, Dropout, BatchNormalization, Add, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv1 = Dropout(0.2)(conv1)  # Dropout for regularization
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv2 = Dropout(0.2)(conv2)  # Dropout for regularization
        
        # Branch path
        branch_output = input_tensor

        # Combine outputs from both paths
        combined_output = Add()([conv2, branch_output])
        
        return combined_output
    
    # Model architecture
    model = main_path(input_layer)
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(model)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    output = Dense(units=10, activation='softmax')(dense)  # Assuming 10 classes for MNIST
    
    # Return the model
    return keras.Model(inputs=input_layer, outputs=output)

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])