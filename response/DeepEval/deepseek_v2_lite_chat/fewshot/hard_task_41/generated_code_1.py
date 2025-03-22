import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths through average pooling
    def block_1(input_tensor):
        # Path 1: 1x1 average pooling
        avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        # Path 2: 2x2 average pooling
        avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        # Path 3: 4x4 average pooling
        avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        
        # Flatten each path and apply dropout
        flat1 = Flatten()(avg_pool1)
        flat2 = Flatten()(avg_pool2)
        flat3 = Flatten()(avg_pool3)
        dropout1 = Dropout(0.5)(flat1)
        dropout2 = Dropout(0.5)(flat2)
        dropout3 = Dropout(0.5)(flat3)
        
        # Concatenate and flatten the outputs
        concat = Concatenate()([dropout1, dropout2, dropout3])
        flat_concat = Flatten()(concat)
        
        return flat_concat
    
    # Block 2: Multiple branch connections for feature extraction
    def block_2(input_tensor):
        # Each branch processes the input through different convolutions
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch7 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch8 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate all branches
        concat_branch = Concatenate(axis=-1)([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8])
        
        # Flatten and pass through dense layers
        flat_concat_branch = Flatten()(concat_branch)
        dense1 = Dense(units=128, activation='relu')(flat_concat_branch)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Process block 1
    block1_output = block_1(input_tensor=input_layer)
    
    # Process block 2
    block2_output = block_2(input_tensor=block1_output)
    
    # Combine inputs from both blocks
    model = keras.Model(inputs=input_layer, outputs=block2_output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])