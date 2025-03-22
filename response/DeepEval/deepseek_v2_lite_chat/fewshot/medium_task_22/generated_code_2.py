import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Branch 2: 1x1 convolution, followed by two 3x3 convolutions
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Branch 3: Max pooling
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(input_layer)
    
    # Concatenate the feature maps from all branches
    concat = Concatenate()([pool1, pool2, pool3])
    
    # Flatten the concatenated feature map
    flatten = Flatten()(concat)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()