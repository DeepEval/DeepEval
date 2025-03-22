import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, ReLU, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Conv layer with batch normalization and ReLU
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    
    # Global average pooling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Fully connected layers
    flat1 = Flatten()(pool1)
    dense1 = Dense(units=128, activation='relu')(flat1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape dense1 to match the number of initial feature maps
    reshape1 = Reshape((32*32*3, 1))(dense1)
    
    # Concatenate with initial features
    concat = Concatenate()([reshape1, conv1])
    
    # 1x1 convolution and average pooling for dimensionality reduction
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(concat)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Final fully connected layer
    dense3 = Dense(units=10, activation='softmax')(pool2)
    
    model = keras.Model(inputs=input_layer, outputs=dense3)
    
    return model