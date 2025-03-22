import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: local feature extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    
    # Dropout layer for regularization
    dropout = Dropout(0.5)(avg_pool)
    
    # Block 2: further processing and global feature extraction
    concat = keras.layers.concatenate([conv1_2, dropout])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Global average pooling
    avg_pool_global = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(conv2_2)
    
    # Flatten layer
    flatten = Flatten()(avg_pool_global)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model