import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Adjust input feature dimensionality to 16 using a convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    
    # Step 2: Apply batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv)
    act = Activation('relu')(batch_norm)
    
    # Step 3: Define a basic block
    def basic_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = Activation('relu')(main_path)
        
        # Branch
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        
        # Combine main path and branch
        output_tensor = Add()([main_path, branch])
        
        return output_tensor
    
    # Step 4: Create a three-level residual connection structure
    # First level
    level1 = basic_block(act)
    
    # Second level
    level2 = basic_block(level1)
    level2 = basic_block(level2)
    
    # Third level
    global_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(act)
    level3 = Add()([level2, global_branch])
    
    # Step 5: Apply average pooling and flatten
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(level3)
    avg_pool = Flatten()(avg_pool)
    
    # Step 6: Add a fully connected layer
    fc = Dense(units=10, activation='softmax')(avg_pool)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=fc)
    
    return model