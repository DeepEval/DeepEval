import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, Flatten, Reshape

def dl_model():
    # Block 1: Feature extraction
    input_layer = Input(shape=(32, 32, 3))  # Adjust the shape to match the CIFAR-10 input
    
    # First path of Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    avg_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second path of Block 1
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(avg_pool1)
    avg_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten and fully connected layers
    flatten = Flatten()(avg_pool2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Connect to Block 2
    branch_output = dense2  # Using the dense layer as the branch output
    
    # Block 2: Deep feature extraction
    def block2():
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv3)
        max_pool = MaxPooling2D(pool_size=(2, 2))(conv4)
        return max_pool  # Output of Block 2
    
    # Connect Block 1 and Block 2
    main_output = GlobalAveragePooling2D()(branch_output)
    block2_output = block2()
    fused_output = Concatenate()([main_output, block2_output])
    
    # Final dense layers
    dense3 = Dense(units=128, activation='relu')(fused_output)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)
    
    # Build the model
    model = keras.Model(inputs=[input_layer, branch_output], outputs=output_layer)
    
    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X_train, X_train], y_train, epochs=10, batch_size=64, validation_data=([X_valid, X_valid], y_valid))