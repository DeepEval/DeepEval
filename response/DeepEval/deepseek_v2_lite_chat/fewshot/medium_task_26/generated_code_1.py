import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Compress input channels with a 1x1 convolutional layer
    conv_compress = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Expand features with two parallel convolutional layers
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_compress)
    conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_compress)
    
    # Concatenate the results of the two convolutional paths
    concat_layer = Add()([conv_1x1, conv_3x3])
    
    # Flatten the concatenated feature map
    flatten_layer = Flatten()(concat_layer)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and return the model
model = dl_model()