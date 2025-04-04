import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # 1x1 convolution to restore dimensionality
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the feature map
    flatten_layer = Flatten()(conv3)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and print the model
model = dl_model()
model.summary()