import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Branch 2: Max Pooling -> 3x3 Conv -> UpSampling -> to original size
    pool1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    upconv1 = UpSampling2D(size=(2, 2))(conv2)
    
    # Branch 3: Max Pooling -> 5x5 Conv -> UpSampling -> to original size
    pool2 = MaxPooling2D(pool_size=(2, 2))(branch1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(pool2)
    upconv2 = UpSampling2D(size=(2, 2))(conv3)
    
    # Concatenate the outputs of all branches
    concat = Concatenate()([upconv1, upconv2, branch1])
    
    # Additional 1x1 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Batch normalization
    bn = BatchNormalization()(conv4)
    
    # Flatten the output
    flat = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()