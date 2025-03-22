import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    
    # Max Pooling layer
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second convolution and max pooling layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Third convolution and max pooling layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Reshape the output of max pooling layers for concatenation
    maxpool1_flat = Flatten()(maxpool1)
    maxpool2_flat = Flatten()(maxpool2)
    maxpool3_flat = Flatten()(maxpool3)
    
    # Concatenate feature maps from different scales
    concat = Concatenate()([maxpool1_flat, maxpool2_flat, maxpool3_flat])
    
    # Fully connected layer 1
    dense1 = Dense(units=128, activation='relu')(concat)
    
    # Fully connected layer 2
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    outputs = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model