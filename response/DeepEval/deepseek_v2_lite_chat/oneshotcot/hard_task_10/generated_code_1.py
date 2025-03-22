import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    
    # Path 2: 1x1 -> 1x7 -> 7x1 convolutions
    path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), padding='valid')(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), padding='valid')(path2)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(path2)
    
    # Concatenate paths
    concat = Concatenate()([path1, path2])
    
    # 1x1 convolution to match channel dimension
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    
    # Branch to input for merging paths
    branch = inputs
    
    # Batch normalization, flattening, and dense layers for the main path
    bn = BatchNormalization()(conv1x1)
    flat = Flatten()(bn)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

model = dl_model()
model.summary()