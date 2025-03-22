import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Basic Block 1
    def basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn
    
    # First path in the block
    path1 = basic_block(input_tensor=input_layer)
    
    # Branch connecting directly to the input
    branch = input_tensor
    
    # Second Basic Block
    def second_basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn
    
    # Third Basic Block
    def third_basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn
    
    # Fourth Basic Block
    def fourth_basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn
    
    # Feature fusion
    block_output = Concatenate()([path1, branch])
    
    # Additions of Basic Blocks
    block_output = fourth_basic_block(block_output)
    block_output = third_basic_block(block_output)
    block_output = second_basic_block(block_output)
    block_output = basic_block(block_output)
    
    # Average Pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2))(block_output)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])