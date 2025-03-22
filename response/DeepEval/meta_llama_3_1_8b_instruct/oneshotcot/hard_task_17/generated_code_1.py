import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Global average pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layer 1
        dense1 = Dense(units=128, activation='relu')(gap)
        # Reshape to match the input's shape
        reshape1 = Reshape((1, 1, 128))(dense1)
        # Multiply weights with input to produce weighted feature output
        weighted_output = Multiply()([input_tensor, reshape1])
        return weighted_output
    
    # Apply Block 1
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return max_pooling
    
    # Apply Block 2
    block2_output = block2(block1_output)
    
    # Direct branch from Block 1
    direct_branch = block1(input_layer)
    
    # Fuse the outputs through addition
    combined_output = Add()([block2_output, direct_branch])
    
    # Fully connected layer 2
    dense2 = Dense(units=128, activation='relu')(combined_output)
    # Fully connected layer 3
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model