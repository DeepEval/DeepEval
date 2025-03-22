import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, ZeroPadding2D, Conv2DTranspose, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolutional layer followed by average pooling and upsampling
    branch1 = input_layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    branch1 = UpSampling2D()(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
    
    # Branch 2: Average pooling, 3x3 convolutional layer, transposed convolutional upsampling
    branch2 = input_layer
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D()(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    
    # Branch 3: Average pooling, 3x3 convolutional layer, transposed convolutional upsampling
    branch3 = input_layer
    branch3 = MaxPooling2D(pool_size=(2, 2))(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D()(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    
    # Concatenate the outputs from the branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    conv_out = Conv2D(filters=3, kernel_size=(1, 1), activation='softmax')(concat)
    
    # Model
    model = Model(inputs=input_layer, outputs=conv_out)
    
    return model

# Create the model
model = dl_model()
model.summary()