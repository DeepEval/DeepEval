import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    
    # Branch 4: Average pooling
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch4)
    
    # Apply Dropout to each branch
    branch1 = Dropout(rate=0.5)(branch1)
    branch2 = Dropout(rate=0.5)(branch2)
    branch3 = Dropout(rate=0.5)(branch3)
    branch4 = Dropout(rate=0.5)(branch4)
    
    # Concatenate all branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()