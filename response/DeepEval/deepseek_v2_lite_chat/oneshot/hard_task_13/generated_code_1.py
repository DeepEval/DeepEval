import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Four parallel branches
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        # Concatenate the outputs of these paths
        concatenated = Concatenate()( [path1, path2, path3, path4] )
        return concatenated
    
    # Block 2
    def block2(input_tensor):
        # Global average pooling
        pooled = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers
        dense1 = Dense(units=1024, activation='relu')(pooled)
        dense2 = Dense(units=512, activation='relu')(dense1)
        # Output layer
        output = Dense(units=10, activation='softmax')(dense2)
        return output
    
    # Apply the blocks
    block1_output = block1(input_layer)
    output = block2(block1_output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])