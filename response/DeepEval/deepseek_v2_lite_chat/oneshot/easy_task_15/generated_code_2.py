import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    # 3x3 convolutional layer followed by average pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    # 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    
    # Second Block
    # 1x1 convolutional layer followed by average pooling
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    # 1x1 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(pool2)
    conv4 = BatchNormalization()(conv4)
    
    # Concatenate features from both blocks
    concat = Concatenate()([conv4, pool2, conv3, conv2])
    
    # Dropout layer for regularization
    dropout = keras.layers.Dropout(rate=0.5)(concat)
    
    # Global average pooling
    avg_pool = keras.layers.GlobalAveragePooling2D()(dropout)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model