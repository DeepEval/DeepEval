import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer followed by batch normalization and ReLU activation
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    
    # Global average pooling to compress the feature maps
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = GlobalAveragePooling2D()(pool1)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(pool1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape to match the size of the initial feature maps
    reshape1 = Flatten()(dense2)
    
    # Weighted feature map generation
    concat = Concatenate()([reshape1, input_layer])
    
    # Reduce dimensionality and downsample using 1x1 convolution and average pooling
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(concat)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Final dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(conv2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()