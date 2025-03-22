import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape, Dropout

def dl_model():
    # Block 1
    inputs = Input(shape=(28, 28, 1))
    
    # Conv Block 1
    conv1_1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1_1 = MaxPooling2D(pool_size=(1, 1))(conv1_1)
    
    # Conv Block 2
    conv1_2 = Conv2D(64, (2, 2), activation='relu')(conv1_1)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    
    # Conv Block 3
    conv1_3 = Conv2D(128, (4, 4), activation='relu')(conv1_2)
    pool1_3 = MaxPooling2D(pool_size=(4, 4))(conv1_3)
    
    # Flatten and Concatenate
    flatten1 = Flatten()(pool1_3)
    concat1 = Concatenate()([flatten1])
    
    # Fully Connected Layer
    dense1 = Dense(256, activation='relu')(concat1)
    reshape1 = Reshape((1, 1, 256))(dense1)
    
    # Block 2
    # 1x1 Convolution
    conv2_1 = Conv2D(32, (1, 1), activation='relu')(reshape1)
    
    # 3x3 Convolution
    conv2_2 = Conv2D(64, (3, 3), activation='relu')(reshape1)
    
    # 5x5 Convolution
    conv2_3 = Conv2D(128, (5, 5), activation='relu')(reshape1)
    
    # 3x3 Max Pooling
    pool2_1 = MaxPooling2D(pool_size=(3, 3))(reshape1)
    
    # Concatenate all branches
    concat2 = Concatenate()([conv2_1, conv2_2, conv2_3, pool2_1])
    
    # Flatten
    flatten2 = Flatten()(concat2)
    
    # Fully Connected Layer
    dense2 = Dense(128, activation='relu')(flatten2)
    dropout = Dropout(0.5)(dense2)
    
    # Output Layer
    outputs = Dense(10, activation='softmax')(dropout)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model