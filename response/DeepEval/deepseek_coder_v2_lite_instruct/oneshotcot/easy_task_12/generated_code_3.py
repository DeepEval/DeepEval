import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add main path blocks
    def main_block(x):
        # First block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Second block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    
    # Apply main block to input
    main_path = main_block(input_layer)
    
    # Step 3: Add branch path
    branch_path = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path)
    
    # Step 4: Sum the outputs from both paths
    added = Add()([main_path, branch_path])
    
    # Step 5: Flatten the result
    flattened = Flatten()(added)
    
    # Step 6: Add fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Step 7: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])