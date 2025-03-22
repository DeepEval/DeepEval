import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # 1x1 convolutional layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # First branch: local feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch1)
    
    # Second branch: downsampling with pooling and convolution
    branch2 = AveragePooling2D(pool_size=(2, 2))(conv1x1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2))(branch2)
    
    # Third branch: similar to second branch but different pooling type
    branch3 = AveragePooling2D(pool_size=(2, 2))(conv1x1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2))(branch3)
    
    # Concatenate outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer to refine the output
    refined = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Flatten the output and pass it through a fully connected layer
    flattened = Flatten()(refined)
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model