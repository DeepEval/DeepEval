import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First layer: 1x1 convolution to reduce dimensionality
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Second layer: 3x3 convolution to extract features
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Third layer: 1x1 convolution to restore dimensionality
    conv1x1_restore = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3x3)
    
    # Flatten the output
    flatten_layer = Flatten()(conv1x1_restore)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model