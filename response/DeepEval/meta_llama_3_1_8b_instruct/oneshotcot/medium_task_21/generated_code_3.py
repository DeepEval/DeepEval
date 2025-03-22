import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Dropout(0.2)(branch1)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(0.2)(branch3)
    
    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch4 = Dropout(0.2)(branch4)
    
    # Concatenate the outputs from all branches
    output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Add batch normalization layer
    bath_norm = BatchNormalization()(output_tensor)
    
    # Add flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1 = Dropout(0.2)(dense1)
    
    # Add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    
    # Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model