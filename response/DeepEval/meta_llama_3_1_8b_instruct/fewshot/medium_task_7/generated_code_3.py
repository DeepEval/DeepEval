import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: three sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Add the outputs of the first two convolutional layers with the output of the third convolutional layer
    adding_layer = Add()([conv1, conv2, conv3])
    
    # Second path: a separate convolutional layer processes the input directly
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from all paths
    output_layer = Add()([adding_layer, conv4])
    
    # Pass the added outputs through two fully connected layers for classification
    flatten_layer = Flatten()(output_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model