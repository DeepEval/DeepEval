import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature Extraction
    def block_1(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate the outputs from different branches
        concat1 = Concatenate()([conv1_1, conv1_2, conv1_3, avg_pool1])
        return concat1
    
    # Block 2: Dimensionality Reduction and Classification
    def block_2(input_tensor):
        avg_pool2 = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool2)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        # Reshape the output to match the input shape of block 2
        reshape_input = Reshape((1, 1, 128))(dense2)  # Adjust the target shape as needed
        
        # Calculate weights using fully connected layers
        weights = Dense(units=128 * 1 * 1)(reshape_input)  # Adjust the shape as needed
        weights = keras.activations.sigmoid(weights)  # Using sigmoid to scale the values
        
        # Element-wise multiplication with the feature map
        output_tensor = keras.backend.batch_dot(input_tensor, weights, axes=[3, 1])
        
        # Final dense layer for classification
        output_layer = Dense(units=10, activation='softmax')(output_tensor)
        
        return output_layer
    
    # Apply block 1 and block 2 to the input
    block1_output = block_1(input_layer)
    model = block_2(block1_output)
    
    # Return the constructed model
    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()