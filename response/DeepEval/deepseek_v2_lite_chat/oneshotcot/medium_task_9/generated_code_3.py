import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First basic block
    def basic_block(input_tensor):
        # Convolutional layer with batch normalization and ReLU activation
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm
    
    # Second basic block
    def second_basic_block(input_tensor):
        # Another convolutional layer with batch normalization and ReLU activation
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm
    
    # Feature fusion
    fusion_layer = Concatenate()([basic_block(input_tensor=input_layer), second_basic_block(input_tensor=input_layer)])
    
    # Additional convolutional layers for enhanced feature representation
    conv_layer_1 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(fusion_layer)
    conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_layer_1)
    
    # Average pooling layer to downsample the feature map
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv_layer_2)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Fully connected layers for output
    dense_1 = Dense(units=512, activation='relu')(flatten)
    dense_2 = Dense(units=256, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model