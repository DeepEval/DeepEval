import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    # Define input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define first block
    def block1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        # Branch path
        branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Combine the main path and branch path
        output_tensor = Add()([conv2, branch_path])
        
        return output_tensor
    
    # Define second block
    def block2(input_tensor):
        # Apply max pooling layers
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Flatten and concatenate the pooling layers' outputs
        flatten_layer = Flatten()(input_tensor)
        output_tensor = Concatenate()([flatten_layer, Flatten()(pool1), Flatten()(pool2), Flatten()(pool3)])
        
        return output_tensor
    
    # Process through both blocks
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(block2_output)
    
    # Apply fully connected layers
    dense1 = Dense(units=128, activation='relu')(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Test the model
model = dl_model()
model.summary()