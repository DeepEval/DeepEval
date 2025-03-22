from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, ZeroPadding2D, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D

def dl_model():
    # Input shape should match the CIFAR-10 dataset
    input_shape = (32, 32, 3)  # Assuming input images are 32x32 pixels
    input_layer = Input(shape=input_shape)
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)
    
    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = MaxPooling2D(pool_size=2)(input_layer)
    path2 = Conv2D(filters=32, kernel_size=1, activation='relu')(path2)
    
    # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate=(1, 3))(path3)
    path3 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate=(3, 1))(path3)
    
    # Path 4: 1x1 convolution followed by a 3x3 convolution, then parallel 1x3 and 3x1 convolutions
    path4 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=3, padding='same')(path4)
    path4 = Conv2D(filters=64, kernel_size=3, padding='same')(path4)
    path4 = Conv2D(filters=32, kernel_size=1, activation='relu')(path4)
    path4 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate=(1, 3))(path4)
    path4 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate=(3, 1))(path4)
    
    # Concatenate features from different paths
    concatenated_features = concatenate([path1, path2, path3, path4])
    
    # Global average pooling
    pooled_features = GlobalAveragePooling2D()(concatenated_features)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(pooled_features)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
model = dl_model()
model.summary()