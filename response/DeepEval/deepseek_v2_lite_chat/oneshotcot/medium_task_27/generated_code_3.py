import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: Convolutional layers
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(branch1)
    
    # Branch 2: Convolutional layers
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(branch2)
    
    # Concatenate branches
    concat = Concatenate(axis=-1)([branch1, branch2])
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(concat)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Softmax attention weights
    attention_weights = Dense(units=2, activation='softmax')(dense1)
    
    # Weighted output
    weighted_output = keras.layers.multiply([concat, attention_weights])
    
    # Final dense layer
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model