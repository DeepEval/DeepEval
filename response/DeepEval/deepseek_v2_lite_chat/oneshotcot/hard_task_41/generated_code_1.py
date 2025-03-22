import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    # Three parallel paths with average pooling of different scales
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Flatten and dropout
    flat1 = Flatten()(path1)
    flat2 = Flatten()(path2)
    flat3 = Flatten()(path3)
    
    drop1 = Dropout(rate=0.5)(flat1)
    drop2 = Dropout(rate=0.5)(drop1)  # Dropout after flattening for regularization
    drop3 = Dropout(rate=0.5)(drop2)  # Dropout after flattening for regularization
    
    # Concatenate and transform to 4D tensor
    concat = Concatenate()( [drop3, drop2, drop1] )
    tensor = Dense(units=128, activation='relu')(concat)  # Dense layer to transform to 4D tensor
    
    # Block 2
    # Four branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(tensor)  # 1x1 convolution
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor)  # 3x3 convolution
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(tensor)  # 5x5 convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=1)(tensor)  # Average pooling
    
    # Concatenate and transform to 4D tensor
    concat2 = Concatenate()([branch1, branch2, branch3, branch4])
    tensor2 = Dense(units=128, activation='relu')(concat2)  # Dense layer to transform to 4D tensor
    
    # Classification layers
    output_layer = Dense(units=10, activation='softmax')(tensor2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model