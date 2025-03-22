import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer and max pooling layer for the first block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_1)
    
    # Step 2: Add convolutional layer and max pooling layer for the second block
    conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_1)
    
    # Step 4: Define a block (we need to combine the outputs of both blocks)
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
        return conv, max_pool
    
    # Apply the block to both paths
    conv1, max_pool1 = block(input_layer)
    conv2, max_pool2 = block(max_pool1)
    
    # Add the outputs of the two blocks together
    combined = Add()([conv1, conv2])
    
    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(combined)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add dense layer (output layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model