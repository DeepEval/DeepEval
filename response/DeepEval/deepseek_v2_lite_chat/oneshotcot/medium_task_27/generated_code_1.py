import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Branch 2: Convolutional layers with larger kernel size
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Concatenate branches
    concat = Concatenate()([pool1, pool2])
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(concat)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Attention mechanism
    weight1 = Dense(units=1, activation='sigmoid')(dense1)
    weight2 = Dense(units=1, activation='sigmoid')(dense2)
    
    # Element-wise multiplication and summation
    weighted_sum = keras.layers.multiply([dense1, weight1]) + keras.layers.multiply([dense2, weight2])
    
    # Final dense layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(weighted_sum)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model