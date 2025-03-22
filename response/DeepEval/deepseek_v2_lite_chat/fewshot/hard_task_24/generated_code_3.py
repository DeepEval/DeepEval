import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Branch 2: MaxPooling -> 3x3 conv -> UpSampling
    pool2 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool2)
    up3 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(up3)
    
    # Branch 3: MaxPooling -> 3x3 conv -> UpSampling
    pool3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool3)
    up5 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(up5)
    
    # Concatenation and final 1x1 convolutional layer
    concat = Concatenate()([conv2, conv4, conv6])
    conv7 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(conv7)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()