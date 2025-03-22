import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Reduce dimensionality using a 1x1 convolution
    conv1x1_reduce = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Extract features using a 3x3 convolution
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_reduce)
    
    # Restore dimensionality using another 1x1 convolution
    conv1x1_restore = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
    
    # Flatten the output
    flatten_layer = Flatten()(conv1x1_restore)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model