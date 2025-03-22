import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D, AveragePooling2D
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Branch 4: Average pooling
    branch4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    
    # Dropout layers to mitigate overfitting
    branch1_dropout = BatchNormalization()(branch1)
    branch2_dropout = BatchNormalization()(branch2)
    branch3_dropout = BatchNormalization()(branch3)
    branch4_dropout = BatchNormalization()(branch4)
    
    # Concatenate the outputs from all branches
    concatenated = Concatenate()(outputs=[branch1_dropout, branch2_dropout, branch3_dropout, branch4_dropout])
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and return the model
model = dl_model()
model.summary()