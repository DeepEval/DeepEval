import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Convolutional layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # MaxPooling layer
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Concatenate all paths
    concat_layer = Concatenate()([input_layer, conv1, conv2, maxpool])
    
    # Batch normalization
    bn = BatchNormalization()(concat_layer)
    
    # Flatten layer
    flatten = Flatten()(bn)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model