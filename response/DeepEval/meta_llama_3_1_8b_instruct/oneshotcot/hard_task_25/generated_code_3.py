import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, Add
from keras.layers import Conv2DTranspose

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch: extract local features through a 3x3 convolutional layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Second branch: downsample the input through average pooling layers, process the downsampled data through 3x3 convolutional layers, and then upsample it through transpose convolutional layers
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)
    
    # Third branch: downsample the input through average pooling layers, process the downsampled data through 3x3 convolutional layers, and then upsample it through transpose convolutional layers
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)
    
    # Fuse the outputs of all branches through concatenation
    output_branch = Concatenate()([branch1, branch2, branch3])
    
    # Apply a 1x1 convolutional layer to form the main path output
    output_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_branch)
    
    # Branch path: process the input through a 1x1 convolutional layer to match the number of channels of the main path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Fuse the main path and branch path outputs through addition
    fused_output = Add()([output_main, branch_path])
    
    # Flatten the output and add a fully connected layer for 10-class classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model