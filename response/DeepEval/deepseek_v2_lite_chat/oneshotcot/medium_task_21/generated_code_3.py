import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    branch1 = LeakyReLU(alpha=0.2)(branch1)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    branch2 = LeakyReLU(alpha=0.2)(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch2)
    branch2 = LeakyReLU(alpha=0.2)(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch3)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch3)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    
    # Branch 4: Average pooling
    branch4 = AveragePooling2D(pool_size=(1, 1), padding='same')(input_layer)
    branch4 = LeakyReLU(alpha=0.2)(branch4)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch4)
    branch4 = LeakyReLU(alpha=0.2)(branch4)
    
    # Dropout layers to mitigate overfitting
    branch1 = Dropout(rate=0.5)(branch1)
    branch2 = Dropout(rate=0.5)(branch2)
    branch3 = Dropout(rate=0.5)(branch3)
    branch4 = Dropout(rate=0.5)(branch4)
    
    # Concatenate outputs from all branches
    concatenated_output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(concatenated_output)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()