import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer with 16 filters and a 3x3 kernel
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Batch normalization and ReLU activation
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = keras.layers.Activation('relu')(batch_norm1)
    
    # Branch for additional feature extraction
    branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Basic block for the main path
    block = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(relu1)
    batch_norm_block = BatchNormalization()(block)
    
    # Addition of outputs from main path and branch
    added_output = Add()([batch_norm1, batch_norm_block, branch])
    
    # Second level residual block
    def residual_block(input_tensor, filters=16):
        conv2 = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        batch_norm_block2 = BatchNormalization()(conv2)
        relu2 = keras.layers.Activation('relu')(batch_norm_block2)
        conv3 = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(relu2)
        batch_norm_block3 = BatchNormalization()(conv3)
        return keras.layers.Add()([input_tensor, batch_norm_block3])
    
    # First level of second-level residual blocks
    block1 = residual_block(added_output)
    block2 = residual_block(block1)
    
    # Global branch for capturing initial features
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    batch_norm_global_branch = BatchNormalization()(global_branch)
    
    # Add the outputs of the second-level residual blocks to the global branch
    fused_features = Add()([block2, batch_norm_global_branch])
    
    # Final classification layer
    dense1 = Dense(units=128, activation='relu')(fused_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])