import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    
    # Branch 2: 1x1 convolution -> 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    
    # Branch 3: 1x1 convolution -> two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    
    # Branch 4: Average pooling
    branch4 = AveragePooling2D(pool_size=(1, 1))(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch4)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation('relu')(branch4)
    
    # Dropout layers for regularization
    branch1 = Dropout(0.5)(branch1)
    branch2 = Dropout(0.5)(branch2)
    branch3 = Dropout(0.5)(branch3)
    branch4 = Dropout(0.5)(branch4)
    
    # Concatenate all branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    
    # Fully connected layers for classification
    fc1 = Dense(units=512, activation='relu')(concat)
    fc2 = Dense(units=256, activation='relu')(fc1)
    fc3 = Dense(units=128, activation='relu')(fc2)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc3)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model