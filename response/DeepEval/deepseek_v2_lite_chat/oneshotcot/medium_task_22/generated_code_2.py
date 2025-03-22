import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Branch 2: 1x1 convolution, followed by two 3x3 convolutions
    conv2 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Branch 3: Max Pooling
    maxpool = MaxPooling2D(pool_size=(2, 2))(input_layer)
    
    # Concatenate features from all branches
    concat = Concatenate()([conv1, conv2, maxpool])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concat)
    flat = Flatten()(bn)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model