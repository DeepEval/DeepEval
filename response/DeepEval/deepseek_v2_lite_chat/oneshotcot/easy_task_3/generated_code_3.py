import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Convolutional block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Convolutional block 2
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv5)
    
    # Pathway for combining features from both sets of blocks
    concatenate = Concatenate()( [maxpool1, maxpool2] )
    
    # Batch normalization and flattening
    batchnorm = BatchNormalization()(concatenate)
    flatten = Flatten()(batchnorm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='relu')(dense2)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model