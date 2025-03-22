import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolution to reduce dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # 1x1 convolution to restore dimensionality
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the output
    flatten = Flatten()(conv3)
    
    # Fully connected layer with 10 neurons for classification
    dense = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model

# Instantiate and return the constructed model
model = dl_model()
model.summary()